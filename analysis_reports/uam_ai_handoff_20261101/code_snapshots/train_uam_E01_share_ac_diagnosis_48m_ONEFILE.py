#!/usr/bin/env python3
"""E0/E1细致诊断：固定模型分支回放＋2×2信息/辅助更新因子，48.16896M新增transitions。

2环境×3机场规模×2续训seed×4版本×1,003,520步；冻结源策略，零初始化logit残差。
不是新的一轮随机初始化抽奖；不修改作者环境、四座满载、奖励、充电或飞机LQ规则。
"""
from __future__ import annotations
import argparse
import ast
import base64
import collections
import contextlib
import copy
import csv
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace
import zlib

VERSION="E01_SHARE_AC_DIAG_48M_V1_20261023"
CANDIDATES={2:(0,1),3:(0,1,3),4:(0,1,3,4)}
NEW_COORDS={3:(0,10),4:(10,0)}
N_FRAMES=6
HORIZONS=(0.,5.,15.,30.,60.)
SOURCE_PPO=dict(n_envs=5,n_steps=4096,batch_size=512,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.25,ent_coef=.01,vf_coef=.5,max_grad_norm=.5)
VARIANTS={"R00_BASE_RESIDUAL":(False,False),"R10_SHARED_READY":(True,False),
          "R01_AC_UPDATE":(False,True),"R11_SHARED_AC":(True,True)}
_SOURCE_K=_SOURCE_PID=_STRICT_INSTALLED=None

SOURCE_PAYLOAD = "c-ri}dv_bhkvRHQCgjNykU<Cl35vuqa#oRKulDNolVs=YA$hPMFeDLy0D=c4Myq>h%X-;*Igw>cmLGB~TXC#M;@Fl*J-)&H5V;(X{L8<OaI2~x)7{fE0A(kez31nIBB!Ujr@Okks=BJW>f}JvZPZ(pcD=bYkQ<n9w;Qe8_U&_(_Izh1S*kB=KiVwL&Q?mpJFAWP;`Y~ezPf+cKw<#iEL7UiVtiyIy~8bL(zEH2Y-X&K&bkxXbapgboSm4NP0wUgshNpldIZ|cR;q4mAa`=$@SdH!_wPwAl%f5gLtW;a*P1iMnzOUg9DcP~@6^i9{`yR%T4^siyXy63xl${(EA`qG{O{k~w%e@Fxiz=mX*sXE&32_xZ?>HyZq+SeCFfwHU0JC7mRdNo^`>*MHoV(iC_=CB0q1D(L<MR$7b_*Vm2-AFyIr^89Cj-;s9SOu+*;e&=N6l_N^Q>B*=RKD#nL=}d2rDA-=F;cqH}N&8@tD`jE3((9RQ=-a$pXXTHCEwD|6UQ*)3ICFpA-Y;xX)Pwpm<okJp>Woc4UN?NnXpsW=Di7CKcJbG=-wI`x?rbWekar*g4fba3QNZG2TRzINTIv+26dqJ%|D!lF`h;PYiR`yzn1jk9m@*(aT&^R6<gV!7OOTR0I|;YQUxk#NdOwc<jhl%PS(Yp`DsOXSuTE6sWhN7iy?mYiSL0SMJ4=%m)Fl-(u{v{8h9YjbYXVS{NUoH@4AZH8i4yIjFR!z*Pf?PA+)vCkKwD}c$OJ6|bPU8huR6lnw%m<cY%%jL?VQv}%8@>2sP7>(PU8u-Oj&2e6?EX+BrW(mr19PQTjPH}FbRFG3oHfnQI1J3bExjhe+c4U7}EnnUaUH(EY{>^3Q|8kd{SKMN|1AxNvVQ1U6{kVF=N9r9|9_I)S7|^K!6<`RoklVK9oV_rdxh77=Ik-CjgjU0K%jUd0eBSQsM-T2Zr;8sx>MlTK00|$Fv*!drvm`#b2iCB@)N=M6Il3P)1$v-bZPBdkHr!g-t(BllL=}B~3G{#~zv(~?`OQOz4i3*0TW;AoRIgS_OWJnBufB%(bEsKAQCWabSSrz;Y}RKwtu{>$zB2>MDR7#9XK$_4BxHw94!g&T%`$A%7OX9P<7lyT49E$9t=Me2usP^GY(wa10amd+@4)xm;a1yi3?HxfRa*1K2JJ)`!i?L7=_3ABXfL^lh839WQc|w?m!CSX$nCMSS(>l3VP^plv4*~R2-{n*1M0J|y_>G<EEInYpMh=NfwAVWMLG#HV;HihR;rirv%jp?OUG!rummQhuY!dh!0xI$Rp<mL!7Rr`7;*+XH!uOd@AuD&S|RH{+pI6})o$Z9b7?qbz5}8E5$vC@=K;5klg4geg@H1>W|-G<Vm=zfZi6Kw$`E%tEu1eP5W&~qQ?Ipv2(kh!x$@2rkFNY}`O@ERya|6eN5%RaY2#<+Ds1B&ov*ZT#eWU!fxj0k2s;LNxuh-GB;mL~#`zG+mDyRhiQr>6-qdz0ki}oR)%tO#h<gRr7&<OBD-AX|#MllF7?uIL@9;SzLeUreZ&Mpu_HEmC0msCauyD;zEw^pkRP8_i^FJF)?Rn&C?6=dbCL2riVyXrlv)C@wXMVlCP=Nm0g#x?-j_H@Xi|u-~$6NZBy`tqs^eI(~MB*z8v_i5d;mlUtYI&+gHQP(HPk5z+4TgXGCHqE__0Gw0fq{}Yc%2mzs@odTzXhOrCDDS2+z}{Xxbd%YTvIjr#VX{GnmbOVoWr$c57<jiZ;!=;j`#N6?(!S!Z+{2>@~SwVoHjZquRSLB?%ubDRW5a!h$97w%?13d0L3JHe;OJesMlOpzg=w3x&2iaBi#7*t?tTIb%ZlTBuYz#W5?%n&TO>~VnUuRObiz~|4Omia^s@C<f?@du#t8SNdweRi7)u?D`k*%sz9G)cV&0B2#l*ROJrgxk1yl?(7Nw`xB32G+0cG?a(?~V-L;j=Ypc)KZ+-H^lMAdC5NL+%1*EqAcvHy~!T;&+y$hFE7v6<Gu06fMI)(1M^Jh1I=XX?dPqYig#bTw3tRSkljq_ivU%9uw`upxjpRnG<kfBN*|8u(6A9|mCj30S^X7lzce1ytbtyvHc=G%obOtuJIt#E8X`w$0PwOFgU<pNT+Ib$tkrC$U8DioW|%AyPFJX5b%;VmtX@*XH`vDvm)5KG-Cwf*kX$rRfFSOb>GDHK4oZ1Y_n1IcbS&Nl%eG(0C2oPi#w2=|06N6DcCf*Mxvn~H_b+y4`<+46P&u-yIRPG3!M(3O_InrzI@@Gl4i%GjGOaH<*~FC!i`=VV|rCiMOJtDSp~?mh78DHTSgVy#dtmE1<VAThp0-%U8R&H`-<+X%!Es&0*HC$;?&cV5EQ&dVxH1uWIi&lg*TR)<RuZH_hk^Fc(jgW@-xN)woEUTxrPQ$Eab5v7dI0udEl&O!smdTc@M3B?7sCmv{hFH~v;-muU_9%x@A70-tKG5;bjzN!KRW&tFyG!H}(2!tJRTQ&l`>)&=Beic^OsH@W&+=xUJ&YW4l_N{m8u7jeUv-arT`d=S-Z-463u^#BnC>o<A-OG13o`2XklH(V}D!bKokp@-(Sc-)<r>jy7ofPO*1kniAQAmbZVRccq_r3YGrmQ#RNs?4#wX@n<8H@GGzW;!wQnysddJX3dAn8rSSBzG$^-@OX@&kogsMwk>Fp3_FtAlo{(Zd!v#`+EnMFq?+h$e>(q7^Mdi2np@^3Hs-dF}q%^ZVZCmwtG1bMu40bnkukx8+;izuX2)?ylbU9{$<E=6_rM$WhR(pE+T(T;x<<-ttWRy{NY`w>e(}9lsL(BS_GuU<C4zS|LCtm6sMNhP_mG7}fzXQmYFh?bN_YnqH-R%2~h@=X?>Uqt7K!BBwcU$>3*G1A#t!06pPM4eX&KI(=Zqr4=+gw)Bf^$KW`Tyhg5-MbbphVFKA`rF{zOBOHzIt6v&)HvjPP#_fw?GMgezi9KzwFO?iE6i*aL$pMUKOv+H<)ZOS=AK1p}r<>=lY<&5acm8)qM?v{cACw7`fTRc2V>+_`NRDg(wR_<Yz_9?F8{d8p0QN3@10@<zgS<8$HrN1WSa&fa(xR>+TIv+^N3&?#fl06_3(EG1*8Tx2dXf7;s#G$iew4FfMP?Mc!X^x&v~lH&wMWbAUtR+8?VbOwd;M|uoA=bk4Yo^`P_Gq>JA-RaZh8-%Y@GY6?YnD_KiT-~Pu)jn*8jK?Ej1i%cHDq?7VOa%fYg;1D@3c>kL)tIRT*Hg53aIr=Yic4dj_58HA@rV-#E8%|Jmk^OB+|8ZY;kKg^U>av+RdM^6%P5x@F|G^wA4}jq1kfogLgQ^(+{Ggyd6z$wTk_!;NRp`iH%~b$bEdCS#id_SWSs^RxQ-6_79%D1OO6Q;iVQA%jjXuB5<FCt-dp<iBW{Id1kMW~vVF7rl;KIonuzNsxUWVc=lC@!+%W<vZS$KlFWgK|EzPh5+5?H07ir@(lpe59Ay!O(o2?G!I#Uo&t3Odra21?Y=;Bb9>3wmVFm!tR8PfJEbg8Um>NUx-MIq4fM?;8yUs2(J58Ot7vNrGo4Dc>^7O!mMce_#ag+(U{%fTuE5}(eZaL6&d%CWB2rgN|7sf7Eh}fksyE#^7(;VOG$Dz&aOBXQT}Sjra+UnQv~Rt>=kU?JhYlV-S|F;ir*L@ZQQ8cO8u*tCMjUBDv>p)RH5bH-wIZxjl9e5HTMabM2<sVdHjL(@<D{#nRxMH7X_Sj?vJ)`lCpUpOF(!8;9HG@nIF)i$Xk&-D6@u9vaA!GWvR&|vUaiu+Tkce$hCXmr+ove{6nh{$`AB6lE;nhEg-SUfx6z_N6GRWJnP7m!`X^7Bau+A|MzQvBc&|SL8Pp5hYRh#drv_3}19&1zPYq0~GeZ?oP{w2s*gSV>{po$S3hH9^8X=pc5V15kmYOcI1FFG+$?#O1J)PcqdeYRXW(R-7>(@T%p1<l{dawKOclL3!ufbl$flm$m`@vuSN8!k*S_7@9g!p-NvrPI8r~SGrivvt5wC5`=pWY$+u<e>jb&g7J{Ht<T8wZ|BIWJ&kP`QOVu;IKIc#<B?D#cm{O$wc++f-^QrA7n6CjH8yB+xTILaN>JBBLtM%Q-w2_%%g|QPIY${YJL8=AcZ~sWLe8mC-cv{OETRS<Qr{T@r$|jz*x6(`txJtUVfZz^SyHTD?vChmi!vyFf^*S4!;6p;MN4K=L5G>>DQ$NO}CngoS<ZwqnclNw<aU;qOL5{AWf?IP3Brf7kuyu6On#NcIlrbKy+^`eICIJyA)&xg*dE;`L&+<L+rT>&@7#P+CFDlcJAPj`#gvyn7#ju>Zr8H>L*S20ar8BAvX-h0y>Kw-W0~Q5hjbe+QSJm743&bQ{$o^MC4pO`vECOR#bK&f3#UY$1MlauFc2e(lfR=hrt+-|b$xy7uhD@D4O%mV!5-u<@ry;)aW|ky~oFw6i%ggjQ<pkYhSC7zP<Cvt+2YjVQLN3#R1Bz~n>?p^}S;Jgr#Vz$_*Nk}*Udqc%yD&wT?g>(RFiN_i$Qla1%sybnM2t~^-3cGr;z>z#jp<M!Q<H2@TY0}7u<*d*A_Not15__R4<e=TU3zA)wg1v2+^*s!}7mb<qu@L_W+zkSRnO_(MTdrt;+^~ymDD$(Cb5K_&vR>XM=n)Pwo3O{i{``+g*{mg}$O4d>-v9{1pUC<!v%oTR-f@Wc-E;@PjKn?ATrsIR*r=XG{JKd2yfl1$CcyjTacmJm1{kMMet+hw*tUX!=@lwTo6MIU)TPtNcoa90FHgKr#c}h+aq6ITpKKUe?d&lo>T4PTr4EVz|xe~*EUB&CyPPX8Sv19I1zFJ(EDHoj+Ip>6iL(l^Zt>xxXDHfC@{H`Wb(_sg0adK7sq_*O#GzeV7J9B1j<@v_>ublOJ7of<y@@DtFhw$&l@@Xf@{+p^PQa&ET#2HW?4RThAT<Q=N>}PT~G_B7pIoMQvrwZV_%Fe|KbD&l$3aJ*lAJy6mOHJO%q*BS0@=i$6_-a&tWpy$D_B1l0Rf`sSQ^{;r9ZEpBP9-yHZ{hN8n%9aaq}^j+-`;(DU)x{!_dQ2n1lId1)k<f<`FFP+1#VU=49;U=a2^kZ^GK>cIFF_JhclT%#!|f?o$U|O9MDs>Q}+Eurx@B>iUtP(p9|G{puEHmml}K%PRP+v2WUdVFX8lYgtG%6;cVsL6$PV!ZicWdV4kJ<qW&9yD@G+oy4hId^7UzZWF0FI#Qv2>(}3q=s&9KX`tp(1|6B8+U|LTffyJL822KNy`*2$cEEO{vgn2Nj@C)4zTY+bPhiv6dknw0fvkKpF(cercP~q={Q|z?szVcHw62!2Az%BAP$Wv{QZKXYO{f{e~@BamkG9W~3eD<e}ho`-jFIWriN2fO$ME>PHJG1uq1MmK4&UR;Qb=7<OHf!<AgF6rJHk&|;8>`)0D{GJLv0BU;XpBYp63ANI6lZ>&S@>ivVKy?p$P7}l*8ZJ)4;<ZdVCR8d3J73atAFjj{rUE_N0*oc<*&g)u@UMwjql)x6%P#!P1SaG+Vk~h&N&2lyz@V%YP-pWIFrhZ45u^0sc|Qh$&F@n>G7%BzGADrr(9_trDGdZ868fghcg-Ag1K}yH#*^0sW)3WrvY6Smw<L>Ml#uv)M(~kfAbr>b~RO_FLOkIWo|4v#g<cRSS<ki$RCCw$o^s@#txDKGeT*sm4Y@3t;%m<>~tW5b0?nCCxKIprKS@h4Jr#v$z87r_v6XQ>22v$D#3nD+rC7?F*4em>5$6JWOUXi?|bxmfWL~zT)bK_8>*ykG4FkptWvqjR02kv&P}FMi8TC`$xUWQ6Haz?`c$tD3b+S>??w&>{wcd<wF-d3LD&TdW}{Gbaa0nzB#H(qskVZ=0a(p{<iWMIXYY6qSDf@baPh2n=IhOK7aiDK>v!KlBW-%#J9`GUyfc;}UmM5pKY4Ja`_b)<XOA84vyZ*EKkdGA$NTi#^>0=|V1-uQYme6+t!#Y%VEyU4-iOa%v)#Y4@#ULLm7o$bplz*sEp#98*X-Apc2g>sP3KbC(A|{64VTU4GCQ`mn?iO@f?Lqw9M2rD*+x1<1K$gaVlgrul9$==t_<)?JZQbPbHCFnxwT@mQcwEzDvK_^WwKB#%~xt}vZw7Q)Qv9aH^)>>zM9B?Q*T!0fGN`T&I;~Lxup%iJbt&*Rx{{jr&yMqy=6Xo!@L}nbL{PpG>UZ5D6F|L1L(^rG3whU!`<S{;;<ndC52eXczjr+8dN?6M#1iUii#HZC3Yz)ssj3Y6sr+6xIa&L1rDm-zEmoBthMcs+RdjK3atefHoE9VbA_E6V3!`J2Auzb2iP|O#U#5)mNV3B+;$nVbj{)-$j>czj@?5I8&|tfMn{@Lqf$E7X^@W#5V>1-(V^(fB}E6b_V^v|>gA2+AEFQBhkxx}e;DZLcoFt%r?_A-O`=cilC_+D9>K2O)G?=XxKnFKA|rPjnHs3M#{rTWgrNvzH-=^)o^J;-1rUe&+Z~2DRM>PE>Wi*<3~6;{3Zipu=?0xeXha>u&tlmjorcwi3P;{FXD2nz)P2j>fGFz5x~^(W1)MAd#O-eZ<mQW0fguRlyC;h8tIxHf;a|WDl?bTG>7L#?zQ2y#I(SvQ2}d!v&!)&E*rIX~j_omS_^|I8#fA`%uz&0-R;z#%s7>N54Az+4^$c&M?{oHtbt9u#Ub0ebomN`NsZJ{`75t4L@ZIOCeJlpM1ISD;TWqxUXxTgS=kBcw-nmcL9(~31`wnfs?q6<0Ih?pSRgrhy>k5G}EkLHt$L@>Ea4uPF;MLt&3~MJ;pDSY*$K%$^9CDKj-LStgZI?Oxx08DNY3C)<07cvvQ5PmhH*WAq**$-qUFGw>|I7O76$z^K)f>VwDR|?9uJji;G3$BbV`_lc<lm(;ZTH?Ekl0_i<ULzn|M33$&9}T;58>a<<!kalsehE4=lcJuAhGIa*({l}ru?ksYdT*K+Aa)~7bJWu7>|Jnns}G|l_UGNKzIz`3rC*N-6&yiLk>YS@`^BE!#kB3O$`v|R0&Wh;|Z`@`=Y7CATgx(7<3l3)uAw<BX9@PyYiLy#!Y_HQV}%>?gst^0WW5wf4R~!&(|+DM8iyjN}3P;;yt_FUH(#B3hgzNUj#03St<=nO|t&gUg%?6w;-9z2iMMW5o*x++pRMW+okM!hGTJ&iNFr#q}>chkzQ*c<(ryqbpQL4YwtSUd!KvfSGyN3p=}?i+1sCbXYMF$W%Gkiyz}qdnq$kiKiN3_)FEeX!RUeRF)7#k{H^t?XKgKYO-V~pn&5SbNm~=&%N_U>8e+A>g{{SOvREz`fPMPc7-IY!88_jf5MF(aEh~`gYw3K+(gynm1tQQ+>x@)K-<naGLG&5VJYl}XF{0kaI&`+77qR#jNd0SV3ed9!Dj)n_h<A9*CMD=ctCy@*<eAC-8c$6xMfA;xx#F%Bypo6-I{X4mqtXahrl3A3@=l^kE-bJ8?T`rSNlZ~0W&X13fRydDwS%ft?lg;j2oY2w;jn`}1<okWt5T-q09X7$^%KjWv+?`}h^XBU9?0lP2u_M1h1f>AF{U!mWK^^qtX$i8aA)o5b96%Bje8xbxsO<}T9S9PZ1}=HXq@5#%MyB%jzQs9fKPHB#kK+j3YbBWY)&-X5>xKRgP{4_%}+7R3_u9L(*>G(>qBq(3Cw)$(IqNbUSB;AB~hn;g*2j@2>YO0W>sh#^*>1v2R{51b+FGrz_SI}I@a%9fB~^Z1i`U;{jtLs$2v$0m+!#R05kCJe~eBCI8qy+ULogo$JZ+8w{Mvp%$P{A(hOU{z!Z#iG<J`5&uW?yF$jKZ6gRMbVKj6H6lXDx6RPA95u%hnT<g*sKaJ5REZD0k&U&Dzcm5;q%Xhqs*HPr=3uhZ;Uvile^<utWZ@K<B|9+%S9T*)=dlS5a2lo93@QPBsS(ah&oPTlnmc(#%01HKbbl@Pv%<b~Jt%K{+sWOYmyxVk>QAAPKMM$k;m^vb;dRl7GIgwfx4j+8&=pN09gDaRcKw|w~+aRda3f&k%Rh3<d8$mpJX><eM?xB!T7d}!kqgPb7fIMYJpAolk<WhU5t<pj_U0-qK-TqULjb%}TTcuPhti}KQt0o#9VM-j%+En9%Zbn(Zzl^s>zW&lX_vz-ji{7*MH@^R9{fkwFD4)=c7n7-JXP9@anGnlMM#Z#JF+Cj=-Js`f+nfwus$8PK;`nbWnKj)VmDqD3H+TH!*IG<}czB<4youVWW>077z$Rp4mRIKtb5CGrsT5eWu~~aw=&qu~LBF0T3Do7PPf?LQ25NB?sHkOXL{C|)tL&G4C2(kH$mz$9n$C@+bE&bOUJ04e7xGFFR}%SSskQh**AuJtxlXeqYQxhUB94AzP#BTvm#FF+X!XkIyn(iOUTuG|5j5+DUmx@>yZj1am}xiqPN5oLgkK^A(SONdfiqzeL^h#kHgX4p<7&4Ll@KNT_Mv{s&}FgQ>a*&XY9KAqwM9<FU)<0P0L<ZX0bxOpl6b&v5zndU-Gpxc!tG-;V3)>#oNDB>Y@h~C+<tQgiR7*$uM=osl5Oq<W^)`7J`~Jv!s@Z6j#s*_sfAWsFOFZ@dhx+pi$<7NCS<jGS!{s}p|wkzK2}?AYGUsNjEu<J(9IAs*W=@K&_Y_7RmP>>&9W)v3b0V+f?q%8|G5>d!#?OmMPJ$w=-$Gx9Trv8r1d2qmmKYg1gH2X#2-ru3TktL{o0BmBRCi0%LG8gCg!5AK!JkB)<F!2$vO#Vs5z8ifly&z<%c@$+2Qe_IQnAE20Yn|Ws)s`5GHXk4pTvF)^M|Dv2Tx~-Gw!=Xs2+qz{<v;-sBL%WoSActOZNkrsWhS93MF>4p#vTnq<CR(;><+wVp6GowWBm#1?BP=S<2`i-m-dCmgeM2}-T*WZ0R)z_lJ3BIx}wtiaxaot@eW${xnc$!Z+yEnu4cVsjt+-n_TmkKxf^E?XETZH0h|G5YeA|7?C3H7)23HwQ;k#?6$|*<ljzT0V<nQVa;x6!sk1l;ly#9_D4MB@A`M>`uCGc;_na>*v;1zhk%gx_57TPp+tM1JsZ+(=sYomW9HH$Gf`Ny>(&z=3DF${bx@<WoBJw(Sr(`x0W})c*7~zI~c#ZH1C#<S?r+x2Y;5gkzU-#ILY8PwrI(bd{h#QnnDBg7pCCLF>}*GkJIrGLeOAGo2h}5SnX8dWaZS<04fRZe~x;m`_J$)l2u7ht(h2R+yrH%u?BkXvu&QoL5e*;G_9S5_&as4+en<bV<sEy-Y9#t*XOt)XjVhJh-w@_cwN)9&~kIVn`nX3ZeHo$TCrmttWW_?u#5|8v;fz#K#TxA<YOVyN<4@z{6;nV6^lDBn=*@K_}VZ`n>}HbKN7XB`FShu7*%>8NZcZN^s&I!g$<l3jrHk+rtYHg!R*Twpaq>`)hBaXkvH^dhDz3w7MeOXwol9$ALxZS9lVpIY@DbTipU)BXXUuWmHu{XF@G7jTF9>`2_!^*j6bPlTEu1Ir!aJV<_tTP%-CMW6Mw(1^`7->i3$WmCa>1$9|Gf1!<_h05O{&E0}wC~5oYFNM$ER4Z2=dAl`=0`bi#%>QUQ~<T6%1%ntX4qI@bmm4v+ytAV^66#!`0fMibo&30VDH;}4a$BCg{LhfP!1L849UR_H})x^!7-y9>eF`juH7fcyXvQ+$aMHo1W};+|?a#DGoS?3(QHWvl&NOsWSlm&Wt{SMh&!+~!hmrAS{ZiYn0K%)D3-&3gnb5FrQ}JOBY(ps=+ekg3ojRNjg&>8o&T;o5Bxr3yd$!blt$^Z@-yEpg%+$jll^Q(+wM0|1)NQ)<{q%_&X;Jr$kl9l2CGH)6YfmBlLQvA*@2zN!qbkA1Rn|F)vXpgZ}OysH;CPouNAJQ_IOrEl1+Iq&nEYmf2jjd$ze=0~5&NA{M-d!<6AlC%YP+iGeJ*$b*pDX`d7jsKaPDYh!5UG>^*WsZ|}%s5rei}!mEymFA!WTu75PYp~S*|q1u&ck~T77pz^a%9heSN9yAc77gPa9f!3B;Jxnv3VGE6;?B^a7NMbkv0fJ3;)JyXxc+j;}tlmi1$+iV&#3upCnYoIAVAi7DjbkhN6MZ5$D-s9A+nnBis6)-dTJ4dvvP2dU<{I9vsTNx8c>>JaZDdchdF4%|CtMt$cwC^t+2}ExXqrdspA}D=dFP7bKY;8l%DT@d9%1qp}z?5DF;KxRh0xTUw|A)H@5PBZ7ym>ck^e-@*={tH5Q9AU(()A&JBwqb0|okfy=-e1Yctg|!1UYJ$;F4c_--;}b>(9ApTx@%#qfjZ;0{*rhkT9_M}WX7}S;xbELRTw6K2_Go$I*;$S;&gPqcSbKclJ9|kz+OaC^j;gcv{G*KrZ@@bbUIXUcf4q6_3O@<Lcm2K)xkypKT{$G(P)5jV0w#MOuE62j`}iI2%J<I3)2Da~H{-0WtT>K}p%|FKIkvEU-f<A$&wu4)@tNO@<$8%s7OzE9*y0Bp4^MBr{U$0%K6uc*ey)4_4;v3wy(`yg<Uo-wI8y(?TuRUl_1a^mPQih)XWx~9O~YHh0Wk6&JXYLpm@grp5{r$&J4YDYx*NbA-vN=;#?LVw3mfC+^39Fs=eTNz6(ukm4Eo>SzsXZKy#F1pDQkvX`|KsQX6VPm!XVvRChCBRV}vzGn2ckvQ<HmYi*cJp2(xLmDht9BK~WM)+t5o&F1uQb_Kq;u?7^H+-0F~~_3--=bq?>7(n*VwPJvWZt&pLpx`h5%c%{H(=flpCPJ=vkI0GSb5|CA%oy})lb1XsEk2DCK@V!6K){{+H1N8gfDu7(4H5o~zg6H5~R-lBgNMR9dNH-!8PV}GAqF+#a9!ADx?j4K~WAW8%bsET}k`fB>pkoQ2J~tg%+#w=CfEta+_X6R$3hDe;-Fu&j>_#CPWmW&gH&V;4XO?F?UM;4XtpW=Xx8Ij)`7w$`Kk=}u)U2+)%VS~d2X(_<1wzOXiHEJry)#m(2NOE3vQAL-)LDqx^@B_ds>K$WeCkI>Qe5BoSc*y<q>!jdfqU(dBSyuWI#8o>0mSw~WZ+h^0Z2cWGk3i|FN^46_ivN$cIfSP9-E?3KTd*;HMCd?of9&t1b5)}p8tu+aj;?<D~`yDn0Dvh1&dGj)If%6W_i_PzY>)V<?<=rXHf}PCE}C{G}7B?ktFj7uUI8+>NCT=*D->|bcmioah|>Z(0lq`_raC5M_=)Mdh;!I;hw$HED)bQ@t$1CIVV+B-z30$0A*P`4G+9}_~2^?cK;W-s3n*?O0zAZm$g#~@{B6Wf^>p=p25RnrA&|QI(x{RO5i<AXWBVs-TcCpn|b1CXeG_YN*p`ifUHhwpXqAg?A_gR_{o@nt#B}77=iP1&ObL>>^H9nZHLM3j2lvJnMFhn?~8|hQTPCbc~PDNVakF{4AliIOt#=KfCs6VI34Udy!YtdT|4*jBoSYAFZ|xS^l<Iboy~KXHm}{-c<=zv9DlyDaqTIYZnl@*na<o+j`>Rq&dx)7U&tznHv-*qJ`g3Eb3IWJs^CBIEjlmo_VIvTDsbc%zmK^~cC{3=QO((Q1P(D*TPq7|f(0=IQwt8?Bo_7xJd~b%8BWQ4D{n7W)@$d4^rUVG)ugQXxZk_Vb5H&7<Yqgc!t9RTne!lTcUNz-GYmVxnDf9Wif#B|laW!G^eOXgQ6Uxb0PGBs#|Sp$@GE#bODf$)9ivU#M_=?6P;n8z#+q@BdCvu~I^c#}b2aH3FTtAvxbFAGwP+}YlMge~c$ULFkb}W%RA|gj4e(eONB10ZPRjjs3JP`}KFYno2Mn+AbduZKyVoBwextsp3sO?Do1A7mSYFDLCp6o|IgwPYx5%0c8sGenYWv!EEJ$*-aA0=^QQ1AwM!m1nZGiVGbVR9UApi!zCyyN3`Ktrd*fx*4$y20I&nv)}N}DZx>P0`JtO0jF4|l_o8dIo=qhSp~RQZG~y0O)uwL}2zG7VliJ@dxK{YQ8r53CPqO3r?>e($=~@5m(8K_+jPl1ud()a$qwk<w1q@gTe;yNwCZ#BDR;sxsDu0WMZVMVpG6)<_R_E-LpT3#QCs`L2Jny1x2}%!-Q|lWUKCVMWX1)z=qvZWL-mGPjn=rFt}U2G=h?gq3C&z+{9Ip?6`S((Ej*=a7^MvkRm`^+WpZAZuibW6n@ms!L2@9%Nd5txqQDXkTTQch^?lSzG-M7O4B_Bk%d?{+FeH^IEi-E!@RH+nY%x@`Jg=0RbtZo4zi=h8gv?N<0?*#8}qBSZ=L^Y;OgDgI9l}ypNd7)Ubw~HjjmPtIb-`dfL-u8EfA_`lV^4u=DWYy|3>n$m|=_j_t-b;nUQB&no2j@+TMVtRrN&!Q^!$Zb_!B?=F7FSkqdVIPM3iH&SQnSsiM|2Xmp!TF=(`Qz6};YKaR3O0Za}qpx5al@&#uTm<_l`0ZVOvn^~4`~ad)(snutZ8|xeo}N_o0Mot37Ca1u7}xtpbrf?5!*}je1)T^KpIKt4mse6sL{wKbY_Js+jEij6_jcJLGZ{&8E+PZr{)^1@^dfvc`uOw$1I&Un^8Ov~`TNM)?tKoYIJOrsyv*Hiy!$t|E`a<vKGzST<C%_$Zj+{yC!T-ls3>us{2O_vls-nk&(Irh5ohITZrTZz&aL=0%kPFk8-<F5*A;BnuP;il4F=BW%<$-s%y-y(_C81`VPs>L>G|>}5iSI_A$%T+hmN3KS1Y&c5miacsMhw?juEYZfT(Q*J%Scyg8+>YVcs}s{1{SOTTqSP!m;`2k?F9(Mw6&n{VHjJV+2LD8dBd3O^0?M?)>YGMQKFCSvZtvZJ;2cI#~&zZf7Az^`-eC1nyg=Bx6bEuzS4NEN`)8lNy=nc7=XHQ_h+<>W#=EW*;`)VMKol`RKu8k*ruQ*#yS(Qr?MSj|STvc6gzxrk@UAdvS5jm|gG-+sX12m|5JUsRZKTyxFI|&~3N5fS!`%8u$uGd74zvZgy%V{?b?dh&-jf5i?AWM;Z=PTJ1Todpbxw*%<RD*>6+T;eXKdG!IerC3uGkWmXA)4hhXD(a(LJeSVb1y<3Q~sKi;+H4jYbXgF_6n}->6F-jvTgCSUg5O52T2wy<qQJJx*uUG_n0APFc0U5l3TiwEj*+WVffS5KkTw->UVlva=QMv}5$dG?*emH`5Gp6)w&fAnJaK^z_Ivj%A_eJJwJ#EOC29C92Fhqq85~eOrC%Ms4QA{&^oEbPo={h1FNN6CCJS$1q=%Zc{pDoqY&<gqqB~msJ6U`)jd-lft+w1q=U%&9j^{Z#Oi<D)uZ6nN()SaQ@_yP}*DWg{f2h1-gahOGi7%;rl^F6|#F<4G3#lc9`Jj4%1VC&xcn#ICgdvp)4o1K4mZS`yK@ptGcwS0Qx#=9HeKk&|e>plIm_xnre$HZbV+2-l2J-xN|Xhn}NDm~5Q*F%su5WlHyjRGjp%m}6-7esioQP89doA2M>{Oki515S<P-R}3_uRZ>A7$!(GSYP>4wXMTR2WGvta+y2iZs)85PC6Fsh&Y9`+%^#QVy6m+5qYx#k0cS-TLUE7nt_bl=AF>J#O?J?$R0900GzN}6|An}8ejx6(XYLl00#K2JzD;Mm*2pkf#+{{7q82sr1xZH{q%X%b6>h@hwBzZ^xb=Q?LD&h;DH`ex>$lx5tv+m(Yr28qNbA)gL&ln2%#Qcp`SA$LvSv?3w+Nd)3ecsAOZQc)yv#LoJcqe(1>m@U#@Q6_}nCN74^bwzZRAdGaI|p;p2OJOH$N9-<#i;qW<2u2?7F4k4*uq!<e+6rkyiyRlYRa2(BR;Nj4JQN|1h~ofa@HtR?sN60NvxC8|YWF=b<mV?e#Jq;(NUT*xt?KGU*_vVY&D39L*|tP0*9`ELULsi4<Es20z>^H2VSYA7&;8C$#zb?G5Q!8VAV&fbMfER$3B{B`a{PHI2kt~@WKHS@7_ptD?F3}%N^eOAJq9s5rc8Enh*M*2xh@!KGqWAZBI2`3Uh(Iu#m%bPSqKT%mYRQ1&&{gu8tX=8rIRc-l^VLG<!P>qMg=Y1>{v^SYYjR=ywC2lG&;p}0)pJa2%%baKwY9N%OnP>GaY1p9ynUMXY`0JEFsr*R`Bu>#CI|jD=Nfn5_b%e$p_^6xb3~PC=Yr2At_QI1vZ>m$FL6;40+`O&yB_|!`jli9xd7#Pu5<DwpB+T$_BO`hOX4}@xS*~BQ!6bj(fvK<$xU?-vI~1q=7Zd54xv^NiO_pT2+iZZhVLKZm&>(JuxbxIIeTAvHyVo9jS3a;c!eAM0RRqfjSz&Vq<hgIFt>_C`!(PV6ehrYec0B%ewDnQd320FaDL2VfNYIMjrnrEZP1*PaeF(Fa1@z*oaW2`f<yp}(M<cbgH}?t_xT{{Rx+Q+a%i+x!m1`+ZcDoNc2M!($i-=~r<ngdG#;O};Yw64!>&x&R51&K803W=b{Q|p~d1jGEi+BDe{hxA10H?R;$hhM&u(o2m-DFcjK1%zC4-h#kBLJ5D;Cp0QlDacOD{v)vd0)H3!K&E=14E0T0vm5w!g|Ckn>Vh+X7-++*}T09$0!a_)SQ4+bj9#a!{&HzFxXkbiP%=SL)aw>WnXaMv9Mr!_8&U>9}52An141k;6(W>_D7HYuC(YyJRxlP49Y>mr6a%4pagpj-(@-i#e2vJJtZb~`;$8>09o8~;Ap4<qAw4HSv46wP)~LAec2cDC-o_w>Q`XvDKrVaLif=b-|@}&Nql;C?a`%m_3pY|r(?m;po_V|lf|}>7N=l6fjydJLHTLvdd&IjGOa6KjlcG1c4vEYd3EF2yWUxJVz3yDbbJeh;I#>6?3^nWnU1-RPTT`A0PRuFigf>yQ3xUjMF&$_S7s|EXYX#3s@Pp*yypzWxLTjVV4o4r;)9OR<tjIuNMC#Tj(6n`sOx=#Sv|TJ&tWL8y}K0+t_Ex=!ZNPHz|eFlTj_!n(d8!0+lAG92*-0&ZZ9l0;{Gq8n=u@@_R3RIp6QIbBGWU=`^;Nls&H#&=xrh<PN!DEU<DWvDB+}%BXNINc=h?*6(^JW`EWM%b0aX^H+Q|W7lD~-x8}U_?{D0`8-96XQZR&RF+{yrj399D4)p3@cMSK@wcIM4R3R_pEK<er8y2;`4vXv*Ytj<i`Os2FB8<WjE!FxyS^9o8DrTSxz*?7MfT*otN-mk1wKnk&tqLZ|^Q+n*u3|MyigWI1_q~T3U%bI@nu`<}-u;i)FJn?%<~+@!dTc!XV&mCa@9}rSq31jteDFTt^ABOiiafch?u{8&WHOb6GviXEFFzrx1FNn*QwDLmk>Q8WO(#>NrlaTRRDKjT-*xsEYx4lQVoi80ia3MpB1|iSLdhWfd8atjiqVK!0CE^A{1Jz9vi~A2UZhoHtnLjVd))5Cc(^*2+`*q~JV5r9F8nw#e0f4s4~zFA2F6yB52thKkz6J+Xm%zyvLl!MDU%Ne^1EBY5WFtZwiMe%WLupQDi28D^zSqA1(ksT@aEI-@k&c%Am#@t<XFBi%L4f$xA^gyiD8|GQbT=ZA;tq!3Mlxr784p!LdEzjyMmE*=9-oAc8s17`@cuC6!{|FBObI&KCG2)HK%ZRMXq!YM1*#k%MJbxxWuTkmVlQf7XdAnct15zfde*XcpY@`uLNc8!C%3KO&3+wC<3v-3o`OHHkL0L8EDrr0C8#F(IT$(Z5v{iYv#bgq?RowF*VYB=_ds4Ve{goGJ-<xq|bvG=WhalMti=}5~ROZR9bbUT5yK|!bM~+v@DsB=M1(%P@t(NbaL7Ea&<3W06~S_WZHU6JH>_<l>vfL7r{N*%p0VxaT`vI?q9|uAZtmb$zf(Qb{c>SrVkX|85Lj72eJ{`77I65m>StweKm)0A4}nVf5$uXefRp~?l<pk(U$suB#ECwk}z$fp%&^g#?<d=2j$!*XAh|%`HHlG>ii?j9i&Ho65hE=cjO0@LTf91L&$bUu^FDjYFg$&Z`tL*uvCFqtj75><g-9_lRlH1jO5apFgY!~!^w=GoR-f16v<5%DE&n5NN)Y)G!dQ#MkuzeBr(y$@I(lUJco&yvZP9Kqn@z@CxvPb=@_AABf*y%SvRtg;Cqoy8q++0P%4}gh8K#*<n5WCI28#s6<Ad3EQm-@=~N)kNRu##zfMS{z=JEu6|X<^KKq!R>);%yM?s+0l&F2NSd~49)|)r4dFMVwDd=~f@Yqe)AF`faFkOi`CrSmO7g>{#G#TgeAFQ>+L}X1-&~MZafgY;IGsm`I@)1Q{q{2kHN-#ya5~IhQscEqgIB0|x=fyE}`Grp7wh#nj;FF@#G-W&Fd#`Wi4U6o(`_y~)ZdaW0|4um~q(4$XBl|xXbiw`c4@h0XwyA;d{%BBjV<6|o^K0IRALBc!#9pHXZtw1Z#_-1{tq}Pies0~UKMRVh;h6|*aj$%e5I_lbj~f<M-bxWmjY3%^z9kG=v<LQ=$g{|lsDTW!3TTA>bFr_`{%X{HiN=H@VjHN=ZMtT~qM8W0Nf|-dQzF#*NIy~Bx7OHiJFO-d765-;z*tT!5fWluKF5=~Y!(mntDJUx4a4T@2OIZ4XUhGxM=Q+hEU*$Uu=xXq%AYQ#di35$22Y>m+nAflZu*smo%JtQy|?e8N3I|S-jxSyt9Ld&`&0MRbLfYD{?pAHpQHEP{fig~mXse_4+JRwiPbOHSMPcEKJae65TQ`7hFRST7MTHui;*>ELf5$lcP0$s0!i&$9}k?H4Gi%Q#5rQ>3JmDp!u5^iGwWZhdQX<uFF)`;I~^YHJ-Ew**Db0R!VwW0ev8>Yx}V(Hc<|YlHCDzs(Q+tUL~66BdX*|`rMQ3U3lwnF#jPmFS)Z}y7kRAt9-pFxRhN|Q`f1&i+T#|DA)Je(B{V!zCrpQBDX~|fqkG}lLSx=nPmIcIp)_^a;t@N14<D}*fmiA$X59917t>Q6uRDS+qN+mC1UBbd)|@9fXYxeCS(*;bSYfZAP6x)(gwvN8m^?q3nqj-!lB)8E5tS3V7AN+Ld^*A}iSowA!V|~F9m@4eJNec_@55)^2iMk~z2iMxanken#yj)%=DCYTe%V-ff?3|7VQ>BVOYg}Qmecl!C+CkXXpsqo3Wp^f$K!Rf^_ym!GSlJ&SY}%I$GA@+a&w($M=q^T1!9E-%37m-WxKLA){NpC=f3eCz6(b?U>KO&>y78$nM<lz&bQk_@kA`mxH*lx<W{ON<J<9gTvc~)%V>nbO4RVf&1yLicDt=ov(jLA_qyU!&cgPi^uqj<$f(Uu{Va=JUly;?TqYfzCU$g3E}PkM8iiik6W;=+hrsVAE<AJ57y1wm5hcBmx+@s?W?ZL8(Yk%FH3ew1Ro4{;u)WZ42k>1I4aGF!{HEh}+(ZD>39_oqRx5M!ZK6J{WN;w;+tQexAr-WZiBzj(7|XCA@QP?xuXdD(+X+?ZGw&q!u*C`6x!)Ona*rM!1?cr+(D6ZpOeqBHd1QzZui2EyLcmCyB{X-qB`-K3C+|evyfj)F>7B9w((LGTlsON_&KH;yalLDTJr08Xp@uAK1!gIm)#}F!vZ+Cn%xuJdW=ieOS3!Ca+Jm&x{gPm;TwM~SU<IKaRo&vE8~)lJmOMy~f&u2#vB^N!fzjz#Ih4B@W}<fu^RS*?0E7CuwiLZYK=D3&C3?5T&}QP!7KflS41!G{<e6e14Man;-rCBAFSiW-q+I<AMVyy3<5>_N)F=*WVkF-b6)LGo`Ame1CR9OjRirIaHc8C4D8-Xu=0UMpLKP!Ho{h{pFzM}Tck%vF!H(6v+KE*S_$KBUKMS5>cb3b*Q~2k|KJEJS%?2H*#ql?m%nhk;Ae%kaZ)A5v6YE}FELN%%BHNz|t-eg(z_xK<K8Lv>4yI9aPqZD`5uYNU1m*oEpLX~y-v{POuJ!VqqmYrnLDo%HTK<Tw5mY&fxB;Y5&AMXpIB<T~fqe0TlizLF47ItFb)-J)!+<X?X2mU`akY5Nh4-_88Eej-wpEC0n83v)Yg$8XcD-(RGP9!a0SsKMWH|-=Dm=#kWtj?8mPSp*1u$5N7^`1N=d;>rnd>mvYV@fYQgZwe=y@Jva{LI}S;31Op%}*ottnTuAgR5H%Wh%Y{uk#JCTbj&x}knZObLj#IQ9shk=7gn8X4u$>INVWI^t9oYh>6NvXN%z$8Amqj%Zk%=EZ4V93U4W?p2>$LJ^=@ws?JUkPcSbE%hLtjBq$$GJy#ai|Ck$kPhQfM?l3F!50|gTUc<O#;xD=4c9|SeyoxSpGm6*B)%kSMY&9I)nFlrom$f*f$`o{#3GdkZg7W%U{PDEo@&7kl(*K)^abNGx>6Q$Z=JR{*7p^x$_x9Ied$vn9|F~UiC>PzFA9cv;wwV|M)=>sF6E+NHUCM!*dD2I(A_Y^rZThmz}wzRg{Vx7nkoZN<GwnBdn>R@R7$3Kyfse5T20WX=@x8>6t|Faub1aVkL-AXtazw~c%hVdPQN+vEV_?yZ;<^2*A{U-B}G$($znKR!#|`_g`McOq=U%b8ax-PmES4?sPq6i+ANlisp};Kec-I*Uuq}_hWU8-nub@C?t}u&zC$7DSK4l)r&kH%fITyZ_3+Xlzmrf!0z1H}?hpzZ2&agUw#>mp+5`O;I9E)^5OnvjAGM{Ug#as>pWZZYw!A<JF54Y$W+Y_!kd*bwINguGv&AnrCLK7;pN!6_<&!zePc@Q2Mew)pY&>|+yS382p+>Y%Io(?qoDm!dre*a$x$Hf9bMy4u-o+KXmHy#hz0beIYgUx;q~EVnva58W-|Cd{D&4!b;VMIBw-68oCHX-^sTT#)K_wz9rpE$;9GxMH{7208r)(=bs3nACidlY%MkL)-D=X5opn{><y#|fkUaFBZB*F*FEEE`dJS4N2b`vX*8H=4_dD^u>p1M-YoglI%*!Q7d)qC)G<IlIvOJ>|*N08Il7Gy^Gk^Y>B#d~$|mtZz5S{us%fevuD@2D61&GA{Cmy8fLA*qfm+4Ph=MbUmuI#LEbSPJZxgm6s?%MPdLh{LIEz|<Tv&q@B>X*JE^Ui;N#yj^Ma!wrrt-%}WlTL)*oSLly$QpqRouPoa-WcARNc2GH$v50Zr`47F-6>fE*2AIt3@ps*e7g&TFb_oP8L8>8|eEDNg81EjPS^wjTcl9kjMxjjH;X76h>IZKEx_<f&NzSNE!jEI1Bd7TdvYSi(7mU<ok{LZv)1VfNLH43tGZ>7FxG?C1cyGzR;?{nIL!Ec+0W1>NB<<aecZZ+7<9+caOg&XdCFoZgf29klWHM>%N{_1JE>cd4!pX`hU#hY+(C!|~(*tuiV@fGoQ}uiPb>y&``YmN)0~&O!n_|%EcWYQ~+J^YFx1>v%dTrdQTVngfyY!we6>;isC@Ho~nbdp2wr)tZ7)_3hi{6P|v@)6=MGM5tS5&=LKCPMODs)8Zmvix?n>@woR@CjijVOJsZTI_C%qUO8!1(>_UcM4g*jU{_T(U5rC~f++-YiC#ZWjBq+mIznp4dWByocjae=}646|f~#s|95;hPx9gC~fXe+WbRu_%bL8wlLj4s<~P=-L0tQ43&HzU=1;pF!y8bfk;WT{OCUhq6wI|Hk*h+_bfZoVj-E<f}pc|nnG4U)oi%iLSUi^!+yanU50~LA*!9?!Nc7M5~>~(2PB<tf))ImP^EgO)>eX_TXy)xbKSo_<I4{q2btJvgh_IkaA*W(kWj~*FvplM#}v0%G+x}>mHb>)-t)VQT67qt>a9x{c$x^>+Ugm8W>XWlF)vSyzWl{%_q{)ejA|FQ;?epd+4g*WAZ!yppogcul`mc(uT9^<IL!#0W@-n0XGa*~S}|)Bz<}k_Y?iZUo08GsyB|F8u72TtbhW#B+ZGjFI_?UISf!CK!!Fn2Vf0MT7(SLv5)Swmy6^mP{loh}mb_aJSt=s$^S6LvtzY{y=3Zc7eC$z&toxq4&@+@V2TrBkuWwV=sNC^BT+vu9#T)cLeG#=hRAuA*SL-X!Z9~_c4f*8-akbSq4fV(**V|MPB@P8<+&fCV^}xr7{%<;!CI&;Qxy`vH!_je@SZwwd$V{!(Ss-mU22dgG-m!(KB_2sSERqfjT^fOIt`@}SWpiXd0?kHY46E<LvN94CHU=}8oZ>&}OC6Ir1O2BXPUC9B&<@S*WKWZ5-HXF{b2CP@)ZDIKvNR2cAf^gq{6OUZVSprG|7JVL<a=$~AOj3(t8GG+M~1}EG*(u7?sKCI3RZUYGT!hwx8g*owZqSg+OFjh)m`fcXW-~~j1DZ1mhluQ>=gJ+y73oFaH=Mt4dLf~v&0uus08>|s0)NZCbtozFayZcMdED{3&QZ}CM;zX$L7u_Sfn<<7|$l=Hc+g}<EQlCq=Vj>uQ5Cmg?3`Hbobgb@77(uH2lTN@7EsPQ}2@n`N9i&QE*wFa+<6?`h5M($K9{L>^|f!@gah|>2op6JTA6v^ZhM`nYRTLX|iJksC)@h0<7;)DeJ2@0(Ktn>=_w&++pF!BX3j~UuN;l{k?&#oV@vAJ;DbP6zBTHWP4ck{t%g1oD!~_uFd!{%5!0{Ila$5_TK)~L}`7CsGL)w>}(k+l7-;bc$s0F4#lt!W9pdi&VVT2y|wIp{w34T_A`P!?$MZ5mJnmI#p8>I<i-sd7Xt<03W%>jjl0(#W40QK`)+L!kS%_Zj~Y!b+}h?}a}+DdzS^%ofL~nf>%Le&ZaQK#s!`Z_d8?1y0MwVbSlcuD$YOsIY`|G&F~U=ZHBfa<tc6-)ZKC_YEQUM4g!|(x>drXGNhpwm(9-GmwEe#PY!H)nv^g0Sft!0oCPM`s)7ep>ACWBtUN=f@)=nviZ0IZUSUWw=ZdA)aux8dl?-<UOurOBQ8mWU1ww(?QIwLfW%vrAAy}Nnt()!c;TQQQpSgeMiEmyaZE0t8UTReoj_0neMLKOGmcR`{~IFr+1^~{?oeH>9D>IE;D^m0fea!=l96l+BnLVO9Nd-rztBRKfW9P9<+(0dApcOKp26GrYml1?SZ*zGCBYedu_PmgSJB$F@<=4nwQ6ILTLRbw7@je50GTH5}<@S51p<He>c6<UYr4V#h8V~w>;)?8?$9je$1;R1Gmy*Ikm`=Co|g(7}Fw4cxs>P~3Fk3%#&BK<n1Y9w%Al+<f?Da=u)DSh$ka7e4Sl5TCW(xe28`wqVP+Tqs<J6}6`aPNV=M_EGisexGvBR#;5MoPpaiW!7o;156!Tb0tTdTq8c7sGN2i;Irtc~N8UfmaSD#4C0)GH(fw`|}tGo~`1X8*foQH&&nJF=)2(lcGe#-NmktwBcm8@NcXR#UKjSd8JZyMGas898A8QB?%QV7C+@R11?KBQwIg%8)Qe=K^`s&1LO04L6+D@WiiS@R}k~zIagr~p5N}i@nGZf_uiH34i>0tT%Je~i}@bnN?&#iV#hHS3F+(G-WxZ0mdVZ2PdC1JgZYN2u^SvLrR0R(`G-m$SdOjlJoV0f3WeNack`|8=`-HhZ=KCI|FH4=2E2grc*{4uFWy{x`kvZ7R%Ks6ed5hGWfH_6p0I1aAXT$QbfRKe5m5g}g_v^i{8jJmPgps_CM4#KW<^B7CV{NeqI}9S77D$OVeAWLjd56|X#%c)X66SDqRb?8`C`PhML6TVnlXw=9u<iOFivtVGEnm55dAqceJar-_%hZFaP6>mN^s@LAw|X>nx>3));4-<IM!4zgRdDumh~mjP1+I5@Ve&jX)c8#Ci8VreNMKx7XGKm<H5Qb++3PJxyaLeGY%F8Tq3+wS>dua*eE)xXTSH(t!zBFV|9x$6+J$rDEdLjC6=57nkJ7es(5RN5jHp}uu^e%a;Q><IprcU%@YPkA#ea0B`${Ip(QH=v@#58Q6CFz5$eFUaYAyes|d~4w+A*K(sI_A*xaDJ7%Dc)T!vdj<wH!aq9i|2RHe~_sL1pmtg3hltIo0+Q(g9FSjA$AJW8-4`_r~3lnc{Zz#vi?F(<P-PuCv5<6XThlghGsz<s!t4EP4x%7_iDt(;XQH03>8$To^ul2L=GqBwQ+I$OZkeAibc>P@ef@D;~Tb{i9k*z=>Al=MttK2)<6iv8pxcq>_IEe>(?ky?$~Vv?O$_zN!N<-_H>n0m>-R<4lIS<eg&Vh{``t;Pe;Qk!Iy7jsS5EyFRjS)V~YY{IybRI0nPv-ynnqEv_Zw8$jIUi<B|JX2yLfw)95JK}mVzn8#|g#+<E8UZS@J1HmGjvJD^%d93k7uOm5W$+sZGI3gaI?!2|0gSFI=2`56uIsXX{PLmccw`UEyI*h3D0;A^P3%!$hp^;ODE7NGq{Tu_hR~}SlEb#|{FqKSup{C*)asBbll8@=S$O3zoiYCs<zGPY5(vTC^ZVUvkAoa@3a<}2F9CHL)G<Q?^^v1{4ml^=r!)pCx)?NpA+Q_~Brl1$C3UN2A!HP=WoQBYQhtbof*5S?gd^5Pw;!<!v^*Pj0*G020k6NgWfmGf;c%Bgq!46};U9=w%}^YWPFeugupMprYt~H28xOu%U*$2_nA~H2M=k^`*W}LGwg?{%OEOJ&VG(K}dsl1!E50m##^nCCt=7paJjj^7`*DtH__0&`D5f>H3J7MM$Sv4oBWpLl%fXA@3f>=)86sHBf0z(39x^E;wTVk3rb&VQ%@ab40aAhzJFU*bLb18jKSf5>kPPUi!y4(F32z&TJS?*t{5XRB!bFqw7kKpyHNAxbvlSI6cW|K~ZABK5L)99Ii))Sy5yfB36k9IW->_FqMKe<pYMGtC+?BvM>g^_Mb0&+a*93nF?K$^{nsLi5L1tctv;X>gchJX%eE(IuhkniOU(IhxpTUx_I4ZO&!orAT;LIIv6h!F=oz_}yErKy86`X+OC@?!F$3wd-E1VAh0OElP<1qbNPsN<F1_J6nSXE%#HFPn*vJL)BW5Rr^Sf(My?124kJbxFBq~}(=EAN^v3@;K=J5=fm5%<D{y)bXeNPBST=b2%LOzOe&n|I{_04Ed<ws9*CpXz_>AKveNdM>O3Wro~J9_)dw5?dMl)ZYE#&GpOYet2^J<unGX^Pap1$h~><O83@EclpcS?vtUgp_oS@?*gHFw40skqft|tD^ygT^yGRj=4vej_KmvDVKYWyfr{1}w=Zt~@jC;#ZL1O3$ijAj{J@?n)2}>3Lsjx~XfL=Shn5!@6UZ(G3s)|qMXyU}Oy!8%XnYExso<^m(0KL_*_H?Q>=Cb1k7}9cNj{BO!3n#ECJ&~Nk8*+RaX$@XTk~<h>_gsSu2)xuw+N}kAWV0G0`@^sLGxjM0e<B3nuzqjnh+m24ip;9Z>`@mfs#2^?ICAo{``Nw9cKF!gJ2R#)QAcPx)Oc5z7Ug9pj6ss(U1{lGX!So&cimo-10{Fn`+CA?w60NG+(cXU@W0KD}YQK;<VMn93IfPf+jjLm5_NSSV#X3#ns>=4DoKzD+g$7z-k@LUMi5umqMZ}bb1I5Guj5}b%v>>8XSSv3MKr8kVm9s5E>7IB4O;Ki#bt=PkhNHRHu~tN#qaJ#m+8@m={EX+?x5cS+!7MHB5Dst=>;3<FH-JtaJTIG0ef8dA;`~wn+J8?a2pHDADVyP+7ls!FzUwbL5XsZ=SnkOeO%I?mKt9PrvP+{^R=7n`R5ETtiThK>3GDGr`D*G!dxgKbVr3F=-sKQM84y<NM(l=ulFJ*<#7vN~$~?`_j`?`G3KFo>uKTVLVD<Hc|Y^^qCCd4_3bLKZA59wm`f4Uh7?Tpv=|!T&1*ybq)ekm9F)$pS`1^@0zdITdo=DcmW9WLa{1iyq6tv$^pD?S;THtd*_~LRO`*6F7a99j9>JUGdNH!a)3c0bcI@^<S&3TZN{ZXRjO6s?f;;FpxM$v!I(zwMUMb7*G@3``>_0vtWb2QVg@g+9``F4;p=7kz#n&tV84FEsUgub!M!raOp?2dcQ5~L^Mg+qvkILfldGyOE<WlGt!{6I3Y<G_<MlP3lHJjK(L?S4THHt7MeZ;{)^_=6-^twWW4@?5nL|sU$K<6vkkI_i4Xucod%$fj!fsN70AZj)1cEvEc*02sfDU^c2$0KS6+>emlG_7)M=sb%Qj-thptAAw!TQP<-H*_dCv@c$?%Hr`#VVgcdmbI9V*Y4evW#8({wREZd6lSI=*O5hBH!H>O!v1+@ZC!RF=CqzywD~AnPQt|)_okQ{lUNNlc}&%HJ_y4=tmu_{S!<J&Io`}Y+?F8xYTRaCB^gRh&;ClJ?cTMfVUPaWeg{WG3rF9X!;gz>p9?NfxvOeA9?06jyX7vfAUk#|G=Zk+Ugx%;B>D)_MTjkXBF?^6(H~4`S&d+mVd@G#Xq_1Z-VSqypv18kNKeh85M?v4BE@KuJrml9w_4f=Opoyoh14=IWT%`Jvnq1tb{ncFV2-6x6!Ip>$Txld#UQGX9>sP%>TF(1x$!>E{kxlb5E7NSW~h6Jw^W)lWYxf!EgAoKW~e)pu$A92@@(bumrF3xK-`L`F!M;jV{CTKn^<XiaSjH2|!tLe9fKW0?1I<|1cZ9jqF>o#P}?;#5b#GmbX~q`3ts-k*w8av}7hV?4&ol*$Z15qDrd-;}8@W_oDCv6h&y*?lY*ZX}Z<8Sm5ST(3<!oIHW%&1_eGMHU}oVAU6o)C^6~1F@#BJ%I_6Yyb+pi{hvbruZP^`cvGLjLlleZ>(&(H;ZkU=v(Q*_iY=$6<@!G8H``MS!>l7qt+u-`OxIzY*~$sGJY0sOOAYz1qkP!mA7-s$JLiC`&XFRBPQAWSq2<TD?<=~pDnuGVie%tn4g2&0eUnQ;B*zsFu@i|U%t))K7pTad{RU=^8M2H%UG;)p(VzINTLcW#sP&+<z?Fd2y#x~~)ffCnnNo-=0$$=rEAgn&D|YA-C-Rq^;v#gz?s$_P<|U$wdI{E1;zosMu%d~M2cp};g7bGKSY9&VE^^UoeQMbtYs&ua&wk3VR7wi`0vIOBktyQaK<_x3wpdW{QoAm))Z>49F$GLeFjdpAHHvsqvgQ2NZPtey#d6s#C!=(j0F(d&4-==1h*Z80KZ{z}fAg)ohVnE%yc&u>W|E@@(NA6h({fN)SJoSdB&7~)!P!a^&in!R868D>QfO+wnO4$0(WZcerYbcaQB~Qmw)z%xvC@1pmBf^E>Oa%*t*%}sX{1ZsN^||bQKZZkazP(f6WtS0jzJM9V-bDAf+6)4(1lf0?wC}{?hAIKk~}V7M$45dQ4C!BrSl+D+H=quSQd7fnL~}M?UiJ;Zmn6zjBNvy4+2GGUA3=;UBf1rk2fT^ghtPm5ScFkdwyx~#o;cs$p(JeLYd6ry>xy!NB}JhEQ-mR0_$S%BO{Ff6S%~!K9wm2uZjEh6jl|Wu>K878Ed6wSrwYUssy(MSHk>neMVpjuxQI7vsL=ry{r6!0aO2PJiqF_{oxN!&aXYXwD#<yjc>2>$a~-arF-iF^U(}A+*ly{5}Q!3U|le$`C_ZshDCK3QQ2<=Kb%TKCTgwj2RBYy-^aH&5>*!IV<bv->qwOMfstsE6eC02V(}yPDBT*MK+3?5O05^{$CTWxW&@^5s5O!cZm|}#LqBc>zJ1>zi*#f<><D|Az~kqC0z5W@#@5VJ{ubs@`3FwJ5PY`=Vim1=jZOVX-yrTA-{i6C1$$71ps@KX>^h!M;w+JIl#!``3Da+_azNP}(}CaJT)x(`wLx832ZKBk80N*0mzN_tQFppB1m13WwP#a7R(azvQXcHY*vJXp2>kdP@1t{UD&B=Zu06fMZ0dGxS!g$WEbE`#V_nI-<!q0#4Qk~h#vZVaj>g4e6x&9p%LQD60vo2-Y!;VdfL#fv++J$9c_<@@j$|~|SG!rPwJ?5$+bjSc;w6u|uB(9OlAm^qo|3g1DKCWS4d^Y}oAJ&>wNi6iBbXX|0pvBRU8ok9(72~uqj-fM#P^6VGA&5tt^wJXA1jJgcJQRIA(3sETE%ZCn#F3h<o<fOQ<bM%i>{+wKm}7ZftGbtZ06G`Eqq?9)4=#3N!d)j%0<4-1S>qWwqm>w*{fQ&-jbDz>_%itel(u!)LOskxbAP=SURo-%U+a?+!KwM$y)W!<Ww?oXPXlnX7AC|&i(@W)2cUPFu@d1$9UY?ZW*UGy~!M$(sH@Xv>ZhXxCc&U8;&hdH?7qz0Yq$dNkIYh6qq;Q<4k9EmLj1vT=b4;xC$&%aV;VuU@+cQ<STa~u4#Bq<eb<q6VB0uvpbl;@Px3}vvKB}6G_xl0I;X~Ldv5qqBWo-$7x5w={|5%G(YO^L)k&hx>aoVi4N}@o^4iuiBy-IwipoCb~$oeGb`W>zZ}?VcghvYT+jO0DPBlzqA$G<J3=4nP%`DZ??E~2Lx2z_#}fK(cV0dG8vLuw%Wol=CTWn7wL?XK#*et=q$*rnz$?`7YeMUX`mTIBGoG+rR^`f4^M-1t<q8FE{%Et~TH8v98;wPIfe7%_fI@2#pP@a*#;f5u_rKuJxn_s|rB&3w2vAjB1v-WKO1bRTLgr?HzT?z5LC&jb&T55uw^(jzm;GmFsoP9;%sTC&zl!F}l~;KT98s*v39|>9CN23u(61ptSc9vpEvm)>VVC<l(LlpwYrfbpwX8>>=26pzqV$gj6y-6XgH@0Oswz*Wokhh5C$Hm&QC9=53eP^YVunidl_q+c?J<YcvoT1~%3Q2z#O0)cpB{06G}kbUf<=&r_xmb%w<N|F$5_FHy$Ry!0YKkaAC(Wg`(BI1gWiS_8pz-z_Lgx9?WI`UQnONPblPZ^{xz!{IioGEQzq%t(fvqAx3*9QYUL$mvCUXkCW0NDxC%3S-$bhV?HWjyEms-=6IpFug_~t<Ifd(|>P)c>@+N7-1-TAA=!MNqBfhpdg5aWTcZ+bT4N^*vN4YbjrVwP^FoMjWr3wP<E$srE+ad#UFPPMju;)2_xB2l_77DYjTjnBEo>Bd+1DC!9<BkM>UkDQffo1j@=xjsjEC@a)7+y3s1gJx^SZTS=>!`cl(`<q`Haj)&TJ2b^e!QmRP0l%~J)0pS`q=&>N#|gv4Lid5H&~`EG-XCWY8s?WIwD%hQ{)E*omUuA^iek`g#%s8ae4nJoiE5ENiB%5EK*}GwQ~X+@9liV(Z%9wWHM@Ftxw1VcO2p1n-tI9a&WMqifrz)2<kz{0yainTt=H+$Y}edyC)J(L401{O|cU}f*00v6xEZnQz#$xZLJhJ1_1=+1Q0fTUA&Z%qcJv&W`2UH2T25u+vXeSib<ZfuJ8;voSuGR!cbI?HYd%mo0LIKuBL3_*gVtGPymr;)$VECfWv)p$w`l8Ml#uv)M(~kfAbsEyC^Bzv%{(EaC&5_hT%#N72ESUXZveAU){fJyWg{srTx&MgN}>#ItDE~F=FEO!~0a@zUq73<9NJxfjfOf*3(r@EDi?@w_2GITBAdBIhEd2>vNrE$5G<Py3MIjRg~=D(7|5XsztYiIx*=&-dR{!dfhGUL01U)je;m8MwcJ!v@6wCE4ffS=Bf?FbAL)Lm>BD*N7cgz58^=u>&FTdFs@LDV>q~ab<u^>Qlr?UrP+UQ_nv))-Fpv1WoomXyGmgo$f?@CgRhz|0fa3PmAA)e(Y?9XT1Ljj5u9W=>^~*u-X6uBq(s4h`)rL)SBt1r<Dr~I;5{fAOGzszi_WL4vZ{d>YiLLwGK8P_!;mQel)$Q*c@E`-Def)8k+$rf$fpv0`soK_^F_wz^DodTfd!fB0hRzhP~7gjeiuekDGW2n&*p27E^pkwx4wFM?fFODi*KwwTHZMKS91cib!6CLL&)RAcRId>b%uk_zIw52Y_Pr>!+s@!6G;UO#z%Za>RF6YRlqG>6Wt~o$EreVKp-^RS{F(8M5ToR-O(bd4#%PBjJ0|j$YG~e{+X|nVJM{5O;%wtF|}LfjO|0@Z#O^v;n5YI)V}-vcbIkg{!I+2eC8Y3Ymyd>xf8uYQM2V6>X?OQ^I!H2?hy{|1^<lMvsKx}`(RRe%N}DGVzpeH8eo@{NUICaMutgdaf7Ojj>7ur<5A1j<kCa(x0QymVt!j~y?{Zn6NGJ<#MuIP09zkrpUdvl43@RUJUYOs$t!WrdG<>QUMXUNyUk>UlJE}D;hp?eRBt%^s#{JvM=<f0!eEmMRQg!G(WG;@Q&Zaa*^)WL1;U47AUP9<+(xAZQ}8iNnoB7k4aMD6eMyebi_Kw#rDG1hOgLqW*=V}x^USSME(<G2c6vH3#x?2R4?h?c;7}KE^`XC%=Uq&urq$Snau%wmNFTt@0`T2drz-rXdp6;S(dv{+ZmWe-;vPk^;xX?0n0zW8$v}BHTo%(#4Et1AgFS5ee9{=YP=iM<IWnutp9slPwcc`hRJuqOf#$$+f*2n6!8#%SSV~Y@xD)Kx(iF&Ihf;}DA`3qgi7Y**6KQy!NbE@Lpyx~?!=ESUc_cBCNK^GhnhHh}qlt0$oT7p({5b{%6THG0JdINi>^<}h&(L3Dgr3LYDGR-h@h@V_aTvvpM23whg&!Qlf5v$iX&7ai8tq7ov64&z{y)x|@=7q!5gN%j?>Rl1fdB8H!SdO{sIyR##@^X*((t7*8ZHc*mB4!(a)yS$ir_2wp~R>JOd94p!A2@TlSw5q2&%M%MJAod(6<n%Y?v8<a0Y=v#jI!~0so(XN@JqZXd*L8Q^T%!SvHYDNRJ^rS;<%;GsdCMR}X-Q@ZJHXybjC*q3^>x10O<&P%XB`0E?7ZWd`4obOQbl@R!Rnl1Ypp#K+`JMgTWx7FirAD;iC}|Je%eNQ|;FShp-qctYURSYianJnDZA2DpP3LBMikJTbzs2tmw>pbskWaS;}TzLl1JkERo&X*PcjfzeE26v6MW%IHX9bcEJPEaoU+^(aAhf`4o@n;6Zq70+@^868WEj?w0jB{0Vwd;xf~@x<sjLtO#gQJClijYmKv3m{9;M8<iuEX*mL*rBXR7KTg+I3iadJCcC^k8|*{foEZOqqLy~OtOIMS!0L7sK;oJWkoai3?jUb6<OGn;{u%-;Ih!ggbv*?c#l;OHyH@V(upxz5J^GC0F5)WeZ*!RgU^hxxxhbcOk)7<QHDISApkPVceMB(p!*mbmFx`uK$wsE`)3TmIL=TGN6X4!MJ9;Sh&A2;?-9lEhfQz?44P2_tjLOB)C^h1_+ssV1sGu@gim<~5CH;!MBg3o6&(4PT&o@MCxRT;hL^xJcd!wTiy}aYaUwhdGj;%ePw)knQyGUADI%(4f`kI*q-knnVvWZE<e3EPjaPzMjcC+p97a7#Q<1b7)|$~2F}ZR0&KQALEXz2M8-|%;myg5N8t0SaUk8xl*38I$CIHMd2Or!MFp@OGCqd^XfD$q?F2FqjLncZg`Nsqdc~oK#gWLp6Axq>*Q1%JfgE;Ptk318w@&qBt>?Yt#<7`lpv`@gdCW!b-fTvUVK1HJv8#|rCp{My+Iaf$wqYO0`&`PIpm?MmT`Gq)vQGG9^Q@BC1d|!$$W4kd%2mRq7)^N<3kl)6MjMLWlKZCe4LHk<l^fY!O5lUcQnmB4&pd+6mGSzg3O<HhA<f`cr&OijGMaG&QB`g_}EE-0f&I*dn2Z_uzJ;sQ!{4`WZlQ@7F%*$Xh>2VDa(#T!Yj4#W{0DHxsF<N*&LS(R+G*RJkSr5g7j9|Q?B1|Pi1k{Hja9RA}>{zshsbq+7_@D(wOFt>vEdmh0Y}3SWGm?;^sULrsSi|QG3>SX@rExY~T!Fcar1V{!2Bw?lj2CGmFNB$lWVnF9Cxe`Kgrk|*-)ZE%Bcq%}ipDUR5ylXt%m>^Te;6|pUxrpAJ2)#5^<XL^#L4hOybQiJB50Ic7-YkvDdOpZmZgCY;|~+t_^^Q!<IfnAV|W=%Wt2!T>y?+H@HWb2M^O`MjAm*11$R#aOU9oa3R_MCQ^p?>>0}wqg|TKrdtL|=8D*SOvVb)3<}~qVg~g<SJ>yT74TzV)WU?gLDdHrsX!>C*%RvD=I!%09l~=PUjAm(Tp`5`BVLI6{VzFalUjnPfAJ$rwK|k4X;t75k%m$fu0&#`+2kaVuRQ{C)hD|>x+{pC@o{c|a>UKk}JvPF&ftWOM?Xgk8ro~)<apTVpffu|KWozs)<4gCzyzys(b|xP<a__Nm+D1NjfPLc+KM3%l0|&<+w&8ed<l)H5*?RHTz{T-roTkX91AH8RvZ|br23C$g6N;dh24;>w49~=PfS=<}meBxFh-U>Rj*JVC15d}FapH5LN0`hw;g~-ccutR}*}fD*gy-})6SF1ff#>x22<<|_Z}B-j&PF{ZAPdjwab42J=QQ#25ufnDb9#ISp_mT_C`gYpsnMVEI39Z@m@MlT;=zXwXg<9~RzD#n5?%&VnaB{$ljk^iPEU+zf+s$wCq}8Ir2WY7CzwPfH#$70C&q}lNo0iQH1T+4N#Hp>!8f{sp;$CA!Os|?IbdOmi;ZH(U_pi${fN)VuppDlh(b{lj&i9Hj*Q}z1_hbaC{0^XJ$%ljgq$Vn!8T2eaXDEwhQ&<n;CzrD*Wfurj6Ulh{NOp0njq4lG=|v-HZM327G%<CE(r4;(|9V)aAXu^FqiZQO>j)Mg{h?3u}B<C@HvytGT~-S7Q$T8V>BE=Pw+XD-ofY00S3>RG?&RacH?s<ElDt+9~5LV?1&^#8J{!6=f{<kGaj%rOk@#UD?=x2VvYWn@ra!v-lQC;@sv$`KI0!Uu^^KXB8#X9b0IcAp&a9(S!SG2PZEDD$Ydr2LWx3{3_D*dLI##(h|`ZMSc)YXV)dhvtmAWLgvl-bmqt)$Fv^&tEQ9%sWW|YG^a%4Ycs)L6Ms{#2!=VPx8RGS61#zJuGcv(R8DATC&Wxt``G<oSo-?Cdob<N>lw?LTOd%r{44yNigk#F?f`SY`f6G-IMHNVv3F{I|p&&C#OkL6*e9nySko%nj4N5YM-TRyZ3o@e<oRNxe!*sIzq#_p=3Nl%y7!x!JpEFs;QRLFVb0#}No88ArD9B_-1!nN+KtYDi<-W*_&*;HPBbQTzf=pKMQMpT?B$H*z7rC|JIg=FvGS@f3b7oA4W>QTAL*s{Zi5BpjA$G405(+Y7Y&zoDiq9E5tTR%>%V08NS-$Hys=#w*jL5Y@lEzR&wSzMdJ{5S*j4_3vzg$p|8Jp0=6nxI?NQounbAf`)jx<wwiMhaYW(N};{I!OH3?1ARE{6q~9is#ZAN!ynL&x_C$r<oDL(D!UPrdk@5l46l+a0KJ<H`pi?qNY@#{}OZqBYEBoKt6s0Z@<`PqXC`vH})l#xoqrg%%HHGd@D1om@hA&ft-rI1OJxc+Sv?URf=8&WtmWQZhMs&d{kor3?n9GR~GtvWjt3A5KUpikhfGq;rDb8lE!~X}<bmDo~Ii86e{e5Ll3z7~wi&zKP*EGr^>Nskwyb%tV%x4?aJ5&P<GP<(6y=lbPUz%kL3BI>FG)*L=ckCMGz<ICS87B*oy*$(hu+Yw<ybhK7DB4-e`*%*8{R7sDJpXdwS|-W}}HQqp>9Ni>PNQSlFT5{d4oRIFBKilt-nUevC6w{)yguhiPR_$%JU+|oj=*s63ElE<5P$JW0}g_m9TRTe7k{ykFES}u23H6gd}$kF{K$as_1%t(R$lnX?jhJ_a9nd($s9R~hH{yQ9DB0ODs_{hk%$%SL(N;5`==0g1#b82<i2U@~tDLd<{_W%E!Y_b6-o2>6-1G6tKwA9S(+(ibXE9qV`Qv=~49ig)32osFdB)niw{WG}tRQh%A!5|8~9DJZoZx<h@A0HoQP|c{zis`B;1@?4+hU!tVZ7x6LERe9r1+X}JBY&y*sU8{9D1JhOYHX<_HDjUWaS4P4o2=j%=qV%e$@S-1^a4&rL6@GG<`|;hnL<yT`b^98byqxe%vYsqr9n;7DLpF%mh&dYtGVLBLNT98PD~hnC;fT*Ah+MrzW&L*eOS2Q!ZDcK94;dHfGE&%+l3-@m^aF`t9z2m7udDQyq#GkoWUOaUyrVopTv`p1OJ#CaB)_77w~Y4TRe+m>WyBUaeiuGf3?9Q9S=mh7gd3WQm_IH+L||bYAovxkH)pTdbA(%qp688rI_8%pldQtHl5KLxQ%*g9<k1fSXv?M_2R-zncidrD2lrn-ej`cd{}I3C{L+(3KwTZv05AV^<1-9Ccix6orWrT?Mdy|ZO+tNu6BP>bI?*FEz?DVnL2DNq+xjssUrjfWcuwH=}xmnM=T-S^y>{&lJZ?a$bNu0L8(u@<XT)rj^X|$PdBsG#iLg!MdzjXp)NQ|292gWTR8z(*eNa)8jZTVYANO^ycCt;*rd0gen9-gfdTdBgmsjkUHWXpB1j-J7hHsmJ!SXbV;*|qe3VT3{#ne4jkan`4dtzMpEsVh7~u?_`AG{ZBm%23y>;M>C7mO%3=TO_Zpp=y%?;Nu7UhpAVL@(qVYMwzw|1{Hi<8{cz)^WL)jZ_jyv)uZmT0Z~MX_DrC!`lUOpgwy(!=SrlS${`|Dzc=1{7QEJ>?2;j#g9;q8iIK?jOsY^k^=-BR8s_&v#aug`>q26(H3jQ}1?~fj$9#2De2F<z@0<h~@AgU!Y)zCkSzT6o)ms0^Kr?mHJyimW}1p8#mrvdvwXWdU5^2`R=W+11HfE2CL@n_Q-Q3yif98M6i$g3I48Bukzsk@?{xs3|pi3Q{4U|od(IsuVC1cgmbjh!1Lv9ikjO)Sun`MPx3^o&c?UjufO^DhbI@;SI@&ASi!q+$$PfUCaI_OTw7V8fHZ?n_raC5M_;LLDddh;$A7u@=pJvt0^49>T}pn9)u^a@VcENNSBswSY&?AmBV_%-09mA$&E-$Ll`r7YQA5gLKRk9Yo*M9$(x9_`;j#DZjqa7J-j(a#xu?*JxEi^ghb?wCo_`2^dS|~`zjqy*Y&?Ciw)!2WQ9t)p_x#n(_nvR>e)!k!^@rW%FE{>rcYF8xIe3pn*B*OUFT*g_9<6$B-`zZS2`YJSe+pxULf-A4mp4z}W$m}~_-R)jz>6QAeBwv8Ew+Ef5IcrA!vEGISsXpG3sK48FX`ma0-UJ2&aXQ)Hf@Se-mbO^m2xl$h9!gxP@5cfN;eM==>ExpC<DSxU%joidkHvKrR40bp(JVKM51e3JZvHLjz=lVUV}?)q%}sdhl=RP><n%$Uw{?E6t|DQ)Z!ocy^^qudn7i=LgvqbsEmWH6j{=0AfnUGHl=GiHQ{ArzbNjQkp)VAI0X0r-}cG<ooc)COQ7G)rPydZ7-d$EqkxzFSUcF^v1<Kz-hpH6d}VH)^Vft!krwhXb#N022LxFbkO(iTHh}52#{yr2BWhdmKUddd6n5EHfLyR&a2GzkwEpxy-`x(|+m2ejduzqJrPN$|^f|QmKDvrIfz^r|Prra#Y}+}o2fNgYKdZTj<lc99dj<D`%Y)O=W6^9pzvg}Tv3LF>@5^@>b)cPeWAofa?~QXC_n-09-1KVm^wYIRD=Gy108cEeP;4SGDk7lyBf){#2f+4=_wXzykZY@#*H(Zk{?&<P6HYo42S|L~z4RG;cJ0wJm8TPa7I)j-+nRg4AT|LV;ncd;S~5J9B{-yy{0YZ$S6zg%BD^uKbE66naV{1cMuYBXJQ(apjzWztLj{8Ia>7AnD3ejntPIn)uuBM{W)Q61kKgiESEwnbHTm?3_vA`MS`+oNM$k>EHQ}l38*Uw#-_Uf<8HBY}BO~+Z0|8hm;do_4V%g1PWjg8ONFacoHZl>`QRxtxPo}0{s99QVW<o&Xh4vz%f&N9Y{G*BCble#X0!az$11BO{OAL1#273`bD`SSi6s>*0zP59J1XKq%Ke_?CdF}b_?i&w+UD5DpnIr%v(M}BiGl8sBpG>iUT4FHTB^R@9CZ|<RPunNN!W;9IQ`)O@0XZOIE<BxBUlYN)7}HuA7|e(TWak>rkVa{((THpmOyMP)=&%~aCas3+mP4wKXw|b=eHLSDSwCO}9d;8B$2bWg?S%?x%kHSu@Cuy1Jrc-xS%BrdvSbMdP8h9xDss0iLK&RzjSlq@MekSph{olJRDvmnWW52ecA&B|#>3yf3k=%3_;~ZorS9F^-pB92o?^o2*&CQPt~3v<{nF;G<&7^$6b9M;@@n_;9hCS%GQE6fbM=Fb`=7%f-uZ_<<>%zRKs~6GU)}LK59pgv8`&5El*L-fwGkZUBVn)cGrv|o!>bQX=WOv@*r$BzBx_Ipno%e};ax0Dk#B3-JCnTDR^>K<oSZ!k2Uf^QuEGl~^;JlU>c2MY0TuhyJ9Edo|9JD-4Mec>-#MEf{H1&EGraZu?5xB#Ko=%6g3xDJXxs%1s^5Y$oLi1DUV$NuweiN|iDT|ko+(tFV>x{`lgCW5C<aeq|M3d}sxIzd>R6ei;+Wsu==WGKIyk$x$-4VlK9G5UwR%Zjp^qNrv1wbd`>l@D3JcLrqzO_R2FQrs#@C$jQ!^e?(pt(jJkaQppSZTFcNGs@Y*pysNhJ+T1wY~#$<k;e$kb6LsW2i3Px@0#ptYsM())slG3@}4D-~;4)PW)rJJ4!3G}pu%&qKECX=0p{?0eIc!bzJO>`yUm_|sU)N+6!Pmf*}M**;iRn$LiWB`O3FXXI%>iU>N;;N0y{Dm~!8Lp+hd&@@jeU`xGVt2YU~O#7Jyh8Bl%Hgp!*1Ud~#p`1J>bBOBI+RQ?b{K-yQn@op6*l(ypmz|3Zq-BNioTZoHw6l%BH;;+zE;wg<APWI&Mm-5B+9pW_7m;yLmkLOcz~Uzvcg%aA>NPcrp*7(dYYYyomI~%gY+Kof3KH6pEO3~R!|;Kt4l3b9j>CFJggV+|+^Xggw5{)B4<*U7oizBV#5p;HLL5k}#N>x^AR-P4p;Rg2xa5liSS(5juA=l*w8WNB@~!AcYqwz}3<E?mvCL5P&eZEw@o%q`mrU=t@%;nu${#FBH+xD;0YZcR%a?W*U|(-@nZ#bi9Bs``t%P}}SO|lbjA0-&k7^tP7Jp^^@&oU)(`zfAudVz}i-<)XAPGZ%0)vK;QnU^j6dO8u3Y|KJl1y0{8zO%>R00phVY!bFB^=@Nl^^P~XNSk3q~;z+gL8gpD2{f7Sqp_c-mJ6<^Pux2dx&9x>`D<;`uQREM9HlVS!kc|QxT-zA|`B=*7u;=&X!5Gy?ZX+^<*m{iM~Y_Vr#z`Gl|<fW-GuKqJBJv9}1~DXvBy(vUh*ZIl1MLoN}h7Fd6sB{@`&+ORX7(5CS&F@c$9)fXC`n%3^$PlBn|E$|R5!c%Q$ue)TL_la{?lU!pegy@%aP_g{#8=n{h%ks(KLsuh@et$sX`<7nIfDkqH3g^|G^uV7YTQdnE8HR3y*bZRDK-vUgAb#f8@5eDE<Gytb3a_Jvo0H&TY87G~|jZEZ5#&u(EYpE6G2R&6|KBMSrNZIn1db_(yuUCwDhzYZd9^w?D2ah02tt5Vx!&W>BO%57vY|W*O!Df$+u*EXtZfm|X^Z&JXEj)G<SNgBCjI6oCj)$tQ>gw(rjChy@B%83d*eJqiR9z2y(7bl;48-z|gt80Xh(~s{%aSN$LnII=VF}tqE1u!ccs$`>>{r#V>b`yN%-CbhMu5S$U-dY3&Z$#X=X__`kCEphSxidwr}`i}xObS}v~#lb!)zLl?`G7SbK&`jd_Cvch1z#^p|PLkj(dCMN}WdgZ0j)U-9ud_h|0TUQo4hm?SC^`>~3RuN1yVn+$+LuHzibGFAw09%k?eiQnwg550&+opEPceml`L6(h%r)@dqtjuC0A<=ePJzlP<zor3YFIy^HTO84nm!&(X@bj_#}O?f>Dko1edV@cUmL{PZ(_j^@(7_f$)O@HDI<d{&IA@qwvywn2Vs=~&sUlZB;fCDiSDr7ik;VqI=^GMyYEmfuOCtQrjF+bmvKc0Iy>b-OP}2<K#V-SyNVt;wy?)F*Amxpw9+c30PQD__sb^+(G)?qjSYP*!`-71?Zq;$bga;y4_q>Bnw!u)Py3qr0t~)VY(ZCe^u=-Fr=5<+IIg*QVf<roO%0Ij$kS<C><4{<DBvYMZ&kETO;6x0)j7*8lwW);m7~VRjs1dz<GsI{5I9&Wqa}GXM7d{a0Su|M<VY`Q7XNL?a!4*`sF|ai5)Wt=Gp&FmkQR)B4ET>Xi+A>s8b|58dSx%mFg?4oR2hqyzB@jGf2AZD$1;`-N%?g*jb@qg=m$Hfcw<1=HZ=&GS1*!QpdrM=kDgm5HOg@R>MDXyxx_CXP9oFpipq<GYhF!@81JIw>a($s*ASKbh<}<>X+ZhtiT9mg!=k@{`PWag3JUS!#@E<Ya2`xG4FqO;tzm`KX`bwn<0PZF}#OD@;bwN^$R{F!w<7v-i%NYznuZJmBgHCJzXDH!S1#6!E`&Z>7{MwJw>)nJ*=|33R;uyZ{~9sj>r<72R2efH31<P||Ir4e)G>!3@#?xLc#2EOrx9R<%5NVz~S>!Sw4ERhary`FLGRo;(_0|4i)vx{fdf!2CMy9h`q<PFBKH$22(<;JJq7-7zIgtMIYi$<F4w*Q)bv!2##mx%SqunI)#3=0gAFgRlPW;EQ+WxXc~j90{Y$yp}*Sn3zi8W!4L$b&0n<FPBxT9-9J}A{O`4P7S4_a{GG8{Z1h!_p^BLB<(n}6pNQSlab)FCn35tHQfxRb*wf9*LKSWlwFvjRms^$CKMWWL<)7ka~+&9nDeKgnj<EQgXfiPm|WK^t!9yq$A#PMamkymc%<^7w?kLD@NvKF+Bq?({r<3c=d&1Ssem23{5k4$aQKVn_du;nGwUpebtxKM-B^m$#$sm@X^IR!xa#cjs~gKNu9}ye%Wc&0<i5Jm?liCMqR>G{kA{~5&g#Bf>D2O3t3<;s!(g-b;d;0CM~^)HM07!JZeZTnc(~c&WoJlsCT0<(<2W5Bse2Y;W9)zO>cOu*-T%X9?nv>^|J*o3AAIra<|e+G+V3sP<q=KC1VzW){oc*V5%x$s0Wd$@DszgpnmX(-nBWDYcG~m)0qyMfr0{{?R%*&~XQ|Iu(F%h>5#H7bK6R#hc-e$FltCgce&OTSZ@u-Wnlxd1TqIk!Jz#~hJ!r<DTlx`*KWlQRZHG>CS10`*1~F$Zl|i?o1%E15?yLOe<3AsK^>Nt6vfhQE1yYYto7Jveb*J4J3{|Mzbem=aU+L9tw<kTCDtSN{s!v|nT-_LT3+xg)*u?V8fsWsvu#hr}ttTu#iedlt|BTLWURlEBt1fKdi}!DS`r7`DcW-|ByKnyKL)--Mv8I0N>A(G9bny8H`~UKLcbxV1`}<$Ned}lczW>H2`|o^s>rLGM-~FF|@`KA8Snl2Y;_HKVe&cxIl|MEw@BjBN_J8^zUXCza(f*e&@4xpa_s|d0<qa%?4nBVS)*s)R`I!$ayfXde!$YSx_>;k287E>)*t9fH?HJFIY=`^cfidc^x9Zw)g?Ex@cJLj)?D)etwYjame7fDG&UW1X8*d%F_R(LyzH#e|_x5jmUXrRL^v%ydIQWm>G(B$;`)*&EbWYM-;nBz^|5{lbM=QD=)RpO-Kgg96kOl7&@4tO+Vckq7@=Divt>VqrxTtP4*(J)}){(Wh(bqCNP1)Bw?PZlW**7u#&bQ$xB)NG0q$n~mI~Baf$l;3wQtU3*%!_92?B!GE9)I-NC*0m$nKnbN1y2{8a4pmdwLG$|xojD#)_Pap<_=zf-L-J~$7fEjpE<2pPCvPF`ohZTOABpC?n+NNc}>%{4kau$q;O{#^?OxHmio<5$8VOfL{M{+prj61P7v#S(X4O2Xxv01ch_2h3bgmOe`C~7!y<K<-epGn7nUCYOP{&!$g$`0pgaGVj{o7h+yT+c7ovZRu6sg{9~Az=>FK@kbz6lc;-_1P?L}vz>lGjDxseJA^t^s{O47r3%AxFzd-!~NbEi(IoJJQOoeG{mX8%iV_^p$mC;lf%00;e1ibix*Zs@fhzjO)Xj@TwUm^ilwd%f@=7Wro5A>aG)kMYl$3m49uI~P6q<am93Jee$Cx^!vp!3WB>K#&K0b04aR6g`;XVR_3vRM#ZR@-%z!0e>;B?llDL-S^b2?t!Rv@ilGh!WX(kE=f48Au39>)<AunedmGle9Y+``bp3<isLbjeaQWh0_B6|AIr01z|wXo7j8erI|j^(tDnE`g8x<n0J;n1<<XBYLx44R{J6c=IP26Wu#a50c)kvd_PN@=de^x-UOXQK4%l-s!zyes7h!E>d44^*^eA(^+R){Z`C(tX$*KpORpVsUJ3KtDF+nP*tl;d)W3MeWFc2kFR{df9*)+11v)`+Sv$Ny{FNM%qb?I?aZ2)G~9NpEBR_%H!oa`UlHqHe^*1_mqq(oFuMszICFhp7kH4K1Tfzj6sDr@N0`Wd)W)rzSiPsNQX#IbR&O}G_}eC&bT27{ZmD8Stsr1KaW`X@#Qv86uiohxJ8Y;AAqUA0=QuC4A|8#<w_T3x$DRF{jLhZQ_th=(_8{`c!gw_+K2!_aLjf^oxj>cE-*@u1wM8Exv$(tISe&h@P0o6<KZl;2)E>z<sge;flkZsxQ3Xma7@{B~DdH=4q^>s8>W<z_bzuT=BG1ChOa%5muRhE4ZsU0Gt!^*8^(shcmY7`eT6ZN<AQO{VuMr_Afdtt+_ac_-X+Z*VK`c)}xMOz(KYE2mB}<rvs(6wR8Wj)^Jiw)8^~&T=bh5acO*wd(+K^Xp$@5nZ~b-+K4P{*70$<iGjZYy1EDrrZ8``O70gaWv3TUMDZLcu{_aSPBuM+Bu_xB;-t;3i_*=HKv^vhB~+8w`+oPgQ}P*uj9InZbP#EJ@ihsI%<CR-U<5Rw#nP8DgD-<J<W7{y|12t%6%Tc;4(&sk<0bYSKGAJiC>isY<-2Ru<l25b@672IoGx_SgVIk*zg^iM5m_+Ivrv-w8ZQ;2^!>jpQL*$)vew27H4a(#KddRrl$7lN`x!7i`ro-Z_6x}SXRBwwy<9n!$c27bRZs~^c-{U_3}k0ww>g$JDbsUS8~@WzOy&bisqx}&25S2t#@9&`Q<Mh<dB!sql;u9f`<#&tmYj{z@~-guT&_G$l!&`n>mb`c845%IQRBB*A4J(v9BXYAGfl`!tq!DurFPBm#0%bpFQ`SzxVu7`Rw^2ze8VU1CSYcckH@`#^xS-nBk~b;Zx;h_v6aJa9s9a+>{mVT$K(1d3|el$22Ck+TGsv`FlI{YvTeg+~ua*0bH*D>tH2cf!Q({7CyJqAJwm{G+%{u@xSJ{$UIYh9lf-=wuV9#gaE_pjRG2t3Ug(3!<D({MvW|o#!^M0f%=1YYMKo8dUmraW|%HJy1zRlJuA&oq&3k%^_}lb*ajOX+#e7AwS;YatHB!q<Y2Wr>Ov3RS#Zap822X@<FVwIu6`YiA{>J*93|}Io`-x4#C3F|Ku$dV!~HuITzxVT9$~ARl-_|UFf+n&^4x;Yn(afFpZ6h*e{;#TYka+I-rf)Vy^Cdf?CVt}doAK?FMgx}v<_oVOdWGF={R{gW?6Br6Zg)y%e%&{fA89LN3LDsUbuFN?_In8f4g?g_OiO~_^w^|-DTgd8P(oX_pa_SNJobr4GDQ3#PK1B#D0!_`tcw2vkrwCsScv3-L=|BP1H#EX@xXV1M=1=$Nug)(?qG%n!B@ITy1_21!c9zKW-tiv$+**+BVp&I*vGh-o1xea#gwM<=P~=YPQW_oYlJ7kuHCTPuFq^rliYD<z|pF0n77y#|y|B*wH@9j`fV~>Dfn3wMz6p`c1@?SgHqd9PNMl!T!6i?*IJV25EP0l+(0gEN#6>b7a?Dn75g4>Yx1$!KQUix3(88qxEX5Lv*GYYcxF-1m|ELf~*1JX&>~WF0tB-97bkI9fxoG%l8g`_3^EDZ?rjehBw`P*)<=Z?XvLbh$k0wPABFE!BpWK#jdr}L3uS$@k;%;!cMLbkz1-S4v@XB;0df;Dg^VP;>#nzl{Ty`9Ioa1_KZ<I1Vp*-qWIETG8KDt<?2~#MDKP5v^><n`vYcMT3l*cHi)U~cOOFYWqROy*wIz%&#g`r^mrdk(6Qo79$1`LO^B)8E~n-MGu;<GTP6-W^vwpS*+X9%?MO8PCAqzRY|}4ZHvHaG`CU{=Fm%mk<e<M_G~EO<Gd4*;XVsQAa>6+3aqeORl({S?Zg58J;4B`I9FC74dFH7j5+noyyT8H$^C#-jrgwoW&({~nvp`y=fWJD1Q(sKVzZ)&J6cjcn^iC<4<SvPSh~Q(j=4{Mf4yu(F(YL3O7l%S6gW@W5l`2^4zUZ49Z|uMK#=%D)mDnt^;(w-?48wYT?+|b-a8nn=ZEg~{PR!~v+s5tw)T?sWXH)My9JwTi4;GlFWw`w1Q`K!}zpPFiKU_ZbeDAP$rP&SWrVv}TUJq0yEJhDa9|L(9$?AnEyvt6};p9{OJ*vUFwt9$=N2FZ599=5?I@}cRhUAz<w1o2M23DFKkJ1UtZ+Qf}A2kG6Ei#@7+D_h17hL#fJGC71^AnKI-7W2pVva7Mfk<_9e=99@o2I2`7(BRse&jI-_i6&ifbRn~z7N>Aw*eb_OQ&Ga#fdmIzJBWB<By!X@c8n&25VAdRmN<ZGOLA+wc<7xEU_vkR&q&s9Oss3q72VyTF_ibo=_oelBq0H9Q<B-A~p$>N|raQAd2%i&#a<~@?6@&a+T$|utgs8IF*J|!v)WaoN*<2p5!?#(kzLKEJ@mO&5JvmYj?SH|NpY+B2HuUO%znABDPdm#d2$vN>f9nOjDI9t&Ol8Rr6RRSym8GFQJ)cF^va{uCZ^*Dx`^1{3DZ0(}d)OLhA}8G)XO|sUcPri6DhFxi$r1Nml3#jV`3deL=I~Dz?oPn0?#Vs&1r8raCqRgRX_hbH)mq$yBMF$-)?=6P;=+B`bKA@=T=G7K~+rvX~Uc4%ac+GO8>;PkF&|v``Tt3o1Ytlc&0n1v5-(NpoccQ7RLODrk~QW~5~pq?8z`jqI0S1MQU<e!dD&hl8+!nS|wuEE1)`cT{CD#UC)E!OvKLdmZIiMhw~^3n?{v#VKp6SeILOje%U*4dfFcR-4RZgvO=-U*wV|HYPd-k*J0UWrd=t1w9Ih_GkE`RaV3j?8VbKZ;LJ62Jc#{tKusZ?2==o1POVfER+Bk&t#S`QV0t%PfRYQiVf$5p#=wv2&R;_3B&;H;Up8qjKaIDuT@$UAQ8StSFBOFO|3!E8Okj*C7h95#HmgT9%pGv2t%u4Wk_7)MM~OA>Rpqv+%$#OGFlU!a-rkGaH+s^(tv4HkznajXs(k)QwFBunrN2gcrUl8H?;vfR*ZPpC|{Rf{aqFhO_OF?P^$!^al%;^o17@hEMx`TVvG@4isl#8X!HtnF}XnTnrDjXbY}5A#Y1C7s3n#p2~mt`%QO=V5<^mz>RbwoKPjV`O$BqD39>^Ea*R2*hJ@8rOYXZ!r@Dp~ORUs9g<&!;gp4J%+Lb|Mf(c6t47!#@Dq@K`xr9I&!JzFxSC*vQSfSFG&8?zKFTzQj+B8og0EDQ*GD(bq4we=_EyN{_1&eJVGW6B(g6K@8B9F5y6Pl6vRkg|#vk6mRbt1Tc!X#ScX<o!C#t<nk62?+3xQ&%&38#*+X~vAjC~?ieIH;%0YWl$6?V4Tl2rL@&_jNsGs@O_Qvdrd~+d%T%KoDpFMj!<z=py8v(=1P|09RQ7k=77<jGE0r;)J8RprT&7#1k0zs$h3Wxv~|sIa4TE#5TtdX)yZeH3nZO%XNx|DUxB>In6OMD4xZboVn8*JR^llL{PLlB`Uq5T4cDk9!?l2tA(P3<R~Z+DUBJX#f)W`OjV-L2*FiilY+%eq(ZTjaHBJBtq`WI=O5f4tVlNsGwOoc!4%1Jk>;FGfxi`t6PcS##i&o2LMKI@fNVCVf?(c6*GN{d#H1Nb^0uzYsyp6aDXZ*MkqezFm;n?*A6tT6!HjTfL&XVovoK>Y0g@u4VA>+jFi$hE9oisB>Z>S^8ScVRR|R*^FDj8^ptYeUhOq+!!fC~v!5+#eCyBsV#|0EuoF#>UMF3SmAodh9Y$DM=7PqAj9IRATPzwde1+1%_3upzSQZP5oF}p!%VFtrMlhC+^m>5ivKu1!T7otd+A(?Hf*lD5W+Fa6QCBm_dp)vGoDxqLmmSJk+(!kKmVM#kXA*Q&XA}b`u$cZ;72-}-xnC(!BKkVPG%u;30#tc&}Wg<x=htWnsRWy^JeCP@e+k;4a4_QHh3dDy>1j{lEcI;+|xtZMEQou@-rE-;H@`t(;&<jMTLW1M4Mt~;OC_9Z|7aK+6f(vTEnu?*-m@eX`iVn4NSh6}LG*cKWr;&MPAU!;V2s6c50_GMHmZ*jH1}90F2QWi3H+3tTr!zCOo&X=t$ndIQ6l-Wg$rTKIL!A{4{gcF;=P*LRd>PD$Tv<@8pd{zHl^IVHhHDP~oQWKa@+|Bs*!wia=a@!Jky2=wGzKA*%%s9_!6GqSgXkG#)?#+y45A4$p3a0CwJ|ykqmvj|PdNi^6#;`u&MZ`zf$*Rb4x0iDkuoAvrYL55BJd<rHkpM=>G+Fd#lR?$xg;dk3}%PSbFeT4V?)hmm|mqJyohybQdDbUa#?7^1k^qpp2%YAb7KcKq*z!M9I~k)<qZ5Ti-JIE3YBS0^jLUtZAmPlsxU8Qv|u8R=V5n%!=Ti$j)mrM)>4us&?>o+C?6JtO>!cgg_X$E<ltw5d4wihODKLRG@k`?kmSy0r6P&JV~iv@jCX}~BD8t}ODZ)?y9qTdFrmPNQUw;zMQpHCiDBQWY*xS|=2?sfgY81H+@!D|pc_m=powBAY}haqY5=P$yh4b;)aN)lhpL2HDZIHc9GF6#6l((U!&0mOd9fh02Bjs&Vo3>UgcXRj8&&|cfK|bSTY9O?2ra?YgYJ}tD+KFf^j^RoO*OR!ObQi;MS;fjg=sOvybSG-3WGtzvQIhj6j=sIEoM43WhQ5_hA1$I8w??dSqnqNvHC7Ra2T)|#|i{XZHz&YLtkUQ2G>DQ6om?yf838%3xNctCU(jUOGhXMn}WQUcVn)hrm)K3nUrp=gLNmk+Ok}uC#aUDdT@VFgIg)Xio>!Pssi>WY)}&$>MaLRXoUfE>ktbejv0uGmS(WMgqz=Vd}?p+{{XCod}0"

SOURCE_PAYLOAD_SHA256 = "0238a5e39c68d0625decc976abd7b27e15216293179e1132d5fa2ee5c1f814ee"

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    """JSON缺失值写null，不用不标准NaN，也不将删失均值填成0。"""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default

def save_csv(path, rows):
    rows = [clean(r) for r in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    os.replace(temporary, path)

def mean(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(finite) if finite else None

def sd(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.stdev(finite) if len(finite) > 1 else (0. if finite else None)

def source_snapshot():
    raw = zlib.decompress(base64.b85decode(SOURCE_PAYLOAD.encode("ascii")))
    if hashlib.sha256(raw).hexdigest() != SOURCE_PAYLOAD_SHA256:
        raise RuntimeError("内嵌作者源码快照校验失败")
    return json.loads(raw.decode("utf-8"))

def materialize_source(root):
    """只向本轮输出目录写独立副本，不覆盖项目已有at_obj等文件。"""
    snapshot = source_snapshot()
    target = Path(root).resolve() / "_author_source" / snapshot["commit"]
    for name, text in snapshot["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("源码包包含不安全路径")
        path = target / relative
        if target not in path.resolve().parents:
            raise ValueError("源码输出路径越界")
        expected = snapshot["sha256"][name]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
            raise ValueError("源码文件内容hash不匹配：" + name)
        if path.exists():
            if digest(path) != expected:
                raise RuntimeError("隔离源码被外部修改，拒绝静默覆盖：" + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
    # 作者部分包初始化文件名含空格；只补空的包标记，不改任何源函数。
    for name in ("utilss", "rl_env", "policy"):
        marker = target / name / "__init__.py"
        if not marker.exists():
            marker.write_bytes(b"")
    return target

def _base_activate_source(root, k):
    """扩站只在本cell进程内配置；旧站、服务和状态转移保持作者实现。"""
    global _SOURCE_K, _SOURCE_PID
    target = Path(root).resolve() / "_author_source" / source_snapshot()["commit"]
    if _SOURCE_PID != os.getpid():
        _SOURCE_K = None
    if _SOURCE_K is not None:
        if _SOURCE_K != k:
            raise RuntimeError("同一worker不可切换机场数，防止全局类配置串扰")
        return target
    sys.path.insert(0, str(target))
    from at_obj.map.map import Map
    original_map_init = Map.__init__
    def map_init(self, *args, **kwargs):
        original_map_init(self, *args, **kwargs)
        for vid in CANDIDATES[k][2:]:
            self.vertiport_station[vid] = list(NEW_COORDS[vid])
        self.vertiport_station_num = len(self.vertiport_station)
    # K2连Map构造函数也不替换；K3/K4只加入新站。
    if k > 2:
        Map.__init__ = map_init
    from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
    for vid in CANDIDATES[k][2:]:
        VERTIPORT_CHARGE_RATE[str(vid)] = VERTIPORT_CHARGE_RATE["1"]
    from at_obj.scenario import Scenario
    if target not in Path(inspect.getfile(Scenario)).resolve().parents:
        raise RuntimeError("导入了项目旧环境，而不是内嵌作者环境")
    _SOURCE_K = k
    _SOURCE_PID = os.getpid()
    return target

def install_learning_classes():
    global np, torch, gym, PPO, SourceFeatures, SourceVecNormalize, SourcePPO
    global SourceEnv, EnvFactory, CheckpointWriter, DummyVecEnv, SubprocVecEnv, VecNormalize
    import numpy as np
    import torch
    from torch import nn
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    class SourceEnv(gym.Env):
        """源wrapper外只读追加表示字段和诊断，不修改奖励、动作、结束规则。"""
        def __init__(self, root, k, horizon, trace):
            activate_source(root, k)
            from at_obj.scenario import Scenario
            from utilss.uam_rl_wrapper import UAMRLWrapper
            self.scenario = Scenario(max_time=horizon, person_spawn_file=str(trace), enable_logger=False)
            for vid in CANDIDATES[k][2:]:
                capacities = self.scenario.vertiports.vertiport_evtol_capacity
                capacities[str(vid)] = capacities["1"]
            self.raw = UAMRLWrapper(self.scenario, list(CANDIDATES[k]), 2, num_frames=N_FRAMES)
            self.k, self.frame = k, 4 + 8*k
            self.base_dim = self.frame*N_FRAMES
            self.focal_start = self.base_dim
            self.query_start = self.base_dim + 4
            self.event_start = self.query_start + k*6
            self.obs_dim = self.event_start + k*len(HORIZONS)*6 + 1
            self.action_space = self.raw.action_space
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        def extra(self, base):
            s = self.scenario
            waiting = s.waiting_decisions
            focal = s.persons.persons[waiting[0]] if waiting else None
            od = list(focal.origin_position) + list(focal.destination_position) if focal else [0.]*4
            queries, tokens = [], []
            for vid in CANDIDATES[self.k]:
                v = s.vertiports.vertiport_list[str(vid)]
                eta = float(s.vehicles.estimate_travel_time(focal.origin_position, v.vertiport_position)) if focal else 0.
                queue = float(len(v.person_list))
                incoming = [max(0., float(p.current_timer or 0.)) for p in s.persons.persons.values()
                            if str(getattr(p, "origin_vertiport_id", "")) == str(vid)
                            and getattr(p, "sub_state", None) == "to_vertiport"
                            and getattr(p, "state", "") == "enroute"]
                # 不预测未来补位飞机，只编码当前已经存在的充电/待命飞机。
                supply = []
                rate = 0.
                for aircraft in s.vertiports.evtols_at_vertiport.get(str(vid), []):
                    state = aircraft.state.name
                    rate = float(aircraft.spec.charge_rate_kwh_per_min)
                    if state == "CHARGING":
                        release = max(0., float(aircraft.spec.battery_capacity_kwh-aircraft.battery_kwh))/max(rate, 1e-8)
                    elif state in ("IDLE", "WAITING"):
                        release = 0.
                    else:
                        continue
                    supply.append((release, float(aircraft.spec.capacity)))
                queries.extend([eta/60., queue/300., len(incoming)/300.,
                                float(v.vertiport_position[0])/60., float(v.vertiport_position[1])/60., rate/20.])
                for h in HORIZONS:
                    threshold = eta + h
                    demand = queue + sum(t <= threshold for t in incoming)
                    seats = sum(n for t, n in supply if t <= threshold)
                    tokens.extend([h/60., threshold/60., demand/300., seats/20.,
                                   len(incoming)/300., len(supply)/10.])
            result = np.concatenate([np.asarray(base, dtype=np.float32).reshape(-1), np.asarray(od+queries+tokens+[float(bool(waiting))],dtype=np.float32)])
            if result.shape != (self.obs_dim,) or not np.all(np.isfinite(result)):
                raise ValueError("源观测或只读增强字段的维度/有限性异常")
            return result

        def reset(self, *, seed=None, options=None):
            # 原作者wrapper忽略seed的行为保留，不在这里修复env.reset。
            obs, info = self.raw.reset(seed=seed, options=options)
            return self.extra(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.raw.step(int(action))
            return self.extra(obs), reward, terminated, truncated, info

        def close(self):
            self.raw.close()

    class EnvFactory:
        def __init__(self, root, k, horizon, trace, directory, seed, rank):
            self.values = str(root),k,horizon,str(trace),str(directory),seed,rank
        def __call__(self):
            root,k,horizon,trace,directory,seed,rank = self.values
            directory = Path(directory)/("worker_"+str(rank))
            directory.mkdir(parents=True,exist_ok=True)
            os.chdir(directory)
            set_random_seed(seed+rank)
            env = SourceEnv(root,k,horizon,trace)
            return Monitor(env,str(directory/"monitor.csv"))

    class SourceVecNormalize(VecNormalize):
        """延续源obs/reward归一化；末尾decision标识只供辅助监督筛样。"""
        def normalize_obs(self, obs):
            value = super().normalize_obs(obs)
            value[..., -1] = np.asarray(obs)[..., -1]
            return value

    class SourceFeatures(BaseFeaturesExtractor):
        """源LSTM真实启用；四个增强方法明确以同一focal底座比较。"""
        def __init__(self, observation_space, features_dim=128, k=2, method="UAGMC_SOURCE_L"):
            super().__init__(observation_space,features_dim)
            from utilss.encoding import TemporalLSTMExtractor
            self.k,self.method = k,method
            self.frame = 4+8*k
            self.base_dim = N_FRAMES*self.frame
            self.query_start = self.base_dim+4
            self.event_start = self.query_start+k*6
            frame_space = gym.spaces.Box(-np.inf,np.inf,shape=(N_FRAMES,self.frame),dtype=np.float32)
            self.temporal = TemporalLSTMExtractor(frame_space,features_dim=128)
            self.use_focal = method != "UAGMC_SOURCE_L"
            self.use_s = method in ("S_EFFECT_EVENT","SA_EFFECT_ICM")
            self.use_ac = method in ("AC_ICM","SA_EFFECT_ICM")
            if self.use_focal:
                self.focal = nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
                self.focal_fuse = nn.Sequential(nn.Linear(160,128),nn.ReLU())
            if self.use_s:
                self.event_net = nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,64))
                self.query_net = nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,64))
                self.shared_out = nn.Sequential(nn.Linear(k*64,64),nn.ReLU())
            if self.use_ac:
                self.ac_state = nn.Sequential(nn.Linear(self.frame,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
                self.ac_inverse = nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,k))
            extra = 64*int(self.use_s)+64*int(self.use_ac)
            if extra:
                self.combine = nn.Sequential(nn.Linear(128+extra,128),nn.ReLU())

        def ac_z(self, obs):
            return self.ac_state(obs[:,self.base_dim-self.frame:self.base_dim])

        def auxiliary_parameters(self):
            return list(self.ac_state.parameters())+list(self.ac_inverse.parameters()) if self.use_ac else []

        def forward(self, observations):
            obs = observations.float()
            latent = self.temporal(obs[:,:self.base_dim].reshape(-1,N_FRAMES,self.frame))
            if self.use_focal:
                latent = self.focal_fuse(torch.cat([latent,self.focal(obs[:,self.base_dim:self.base_dim+4])],-1))
            extras = [latent]
            if self.use_s:
                query = self.query_net(obs[:,self.query_start:self.event_start].reshape(-1,self.k,6))
                events = self.event_net(obs[:,self.event_start:-1].reshape(-1,self.k,len(HORIZONS),6))
                weights = torch.softmax((events*query[:,:,None,:]).sum(-1)/8.,dim=-1)
                shared = (events*weights[:,:,:,None]).sum(2).reshape(-1,self.k*64)
                extras.append(self.shared_out(shared))
            if self.use_ac:
                extras.append(self.ac_z(obs))
            return self.combine(torch.cat(extras,-1)) if len(extras)>1 else latent

    class SourcePPO(PPO):
        """PPO主目标不改；ICM只做逆动力学辅助，不额外发奖励。"""
        def _setup_model(self):
            super()._setup_model()
            self._save_hook = None
            params = self.policy.features_extractor.auxiliary_parameters()
            self.aux_optimizer = torch.optim.Adam(params,lr=self.lr_schedule(1),eps=1e-5) if params else None

        def _excluded_save_params(self):
            return super()._excluded_save_params()+["_save_hook","aux_optimizer"]

        def _get_torch_save_params(self):
            states,variables = super()._get_torch_save_params()
            if getattr(self,"aux_optimizer",None) is not None:
                states = states+["aux_optimizer"]
            return states,variables

        def train(self):
            super().train()
            if self.aux_optimizer is not None:
                buf = self.rollout_buffer
                # SB3在PPO更新时可能已将buffer展平；按是否generator_ready还原布局。
                if buf.generator_ready:
                    obs = buf.observations.reshape(self.n_envs,self.n_steps,-1).transpose(1,0,2)
                    actions = buf.actions.reshape(self.n_envs,self.n_steps,-1).transpose(1,0,2)
                else:
                    obs,actions = buf.observations,buf.actions
                starts = buf.episode_starts
                eligible = (obs[:-1,:,-1]>.5) & (starts[1:,:]<.5)
                indices = np.flatnonzero(eligible.reshape(-1))
                losses=[]
                for group in range(4):
                    if not len(indices):
                        break
                    selected=np.random.choice(indices,size=min(512,len(indices)),replace=False)
                    current=obs[:-1].reshape(-1,obs.shape[-1])[selected]
                    nxt=obs[1:].reshape(-1,obs.shape[-1])[selected]
                    labels=actions[:-1].reshape(-1,actions.shape[-1])[selected,0].astype(np.int64)
                    x=torch.as_tensor(current,device=self.device)
                    y=torch.as_tensor(nxt,device=self.device)
                    features=self.policy.features_extractor
                    logits=features.ac_inverse(torch.cat([features.ac_z(x),features.ac_z(y)],-1))
                    loss=.1*torch.nn.functional.cross_entropy(logits,torch.as_tensor(labels,device=self.device))
                    for group_params in self.aux_optimizer.param_groups:
                        group_params["lr"]=self.lr_schedule(self._current_progress_remaining)
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(features.auxiliary_parameters(),.5)
                    self.aux_optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                self.logger.record("aux/inverse_loss",mean(losses) or 0.)
                self.logger.record("aux/valid_transitions",len(indices))
            if self._save_hook is not None:
                self._save_hook(self)

    class CheckpointWriter(BaseCallback):
        def __init__(self,directory,every):
            super().__init__()
            self.directory,self.every=Path(directory),every
        def _on_training_start(self):
            self.next=((self.model.num_timesteps//self.every)+1)*self.every
        def _on_step(self):
            if self.num_timesteps>=self.next:
                save_pair(self.model,self.directory/"checkpoints",self.num_timesteps)
                self.next+=self.every
            return True

    # 固定公开类名，便于SB3保存和跨进程载入，不依赖嵌套函数的局部名字。
    for cls in (SourceEnv,EnvFactory,SourceVecNormalize,SourceFeatures,SourcePPO,CheckpointWriter):
        cls.__qualname__=cls.__name__
        cls.__module__=__name__

def save_pair(model,directory,step):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    stem=f"step_{step:09d}"
    model_file=directory/(stem+".zip")
    norm_file=directory/(stem+".pkl")
    model_tmp=directory/(stem+".pending.zip")
    norm_tmp=directory/(stem+".pending.pkl")
    model.save(str(model_tmp))
    model.get_vec_normalize_env().save(str(norm_tmp))
    os.replace(model_tmp,model_file)
    os.replace(norm_tmp,norm_file)
    save_json(directory/(stem+".json"),dict(step=step,model=model_file.name,vecnorm=norm_file.name,time=now(),n_updates=model._n_updates))
    return dict(step=step,model=str(model_file),vecnorm=str(norm_file))

def episode_metrics(env,counts,restricted_area):
    """诊断统计不反馈给仿真；完成口径去重但不修复源状态机。"""
    s=env.scenario
    persons=s.persons.persons
    finished=set(s.finished_ids).intersection(persons)
    values=[]
    for pid in sorted(finished):
        person=persons[pid]
        # 作者记录与原测试常用的时间分量另列；全体旅时用可追溯的时间戳。
        record=[r for r in s.person_travel_records.get(pid,[]) if r.get("end_time") is not None]
        if not record:
            continue
        end=float(record[-1]["end_time"])
        value=dict(id=pid,travel=end-float(person.spawn_time),source_travel=end-float(record[-1]["start_time"]),
            ground=float(person.time_stats.get("to_vertiport",0)),wait=float(person.time_stats.get("wait_uam",0)),
            flight=float(person.time_stats.get("fly",0)))
        values.append(value)
    full=bool(persons) and len(finished)==len(persons) and len(values)==len(persons)
    weights=[n/max(1,sum(counts)) for n in counts]
    entropy=-sum(x*math.log(x) for x in weights if x>0)
    return dict(generated=len(persons),finished=len(finished),completion=len(finished)/max(1,len(persons)),full=full,
        ATT=mean([v["travel"] for v in values]) if full else None,
        AWT=mean([v["wait"] for v in values]) if full else None,
        ATT_completed_only=mean([v["travel"] for v in values]),
        source_ATT_completed_only=mean([v["source_travel"] for v in values]),
        AGT_completed_only=mean([v["ground"] for v in values]),
        AWT_completed_only=mean([v["wait"] for v in values]),
        AFT_completed_only=mean([v["flight"] for v in values]),
        restricted_person_time_per_generated=restricted_area/max(1,len(persons)),
        action_counts=counts,action_shares=weights,effective_airports=math.exp(entropy),dominant=max(weights),
        aircraft_ever_created=len(s.evtols.evtols),simulation_minutes=s.time,
        missing_passengers=sorted(set(persons)-finished),
        metric_note="源码终点为航空到达；原代码未建模的末段地面不补算；未完成全体ATT为null",
        passengers=values)

def install_closed_loop(k, fleet_size):
    """只替换飞机供给闭环；不替换源step、满载调度、奖励或结束条件。"""
    global _STRICT_INSTALLED
    signature = (os.getpid(), int(k), int(fleet_size))
    if _STRICT_INSTALLED is not None and _STRICT_INSTALLED[0] != os.getpid():
        # spawn/cloudpickle可能带来父进程缓存；新进程必须从隔离源码重新安装。
        _STRICT_INSTALLED = None
    if _STRICT_INSTALLED is not None:
        if _STRICT_INSTALLED != signature:
            raise RuntimeError("同一进程不得切换机队配置，防止环境补丁串扰")
        return
    import at_obj.scenario as scenario_module
    from at_obj.evtol.vehicle_state import VehicleState
    from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
    OriginalScenario = scenario_module.Scenario
    candidates = tuple(str(v) for v in CANDIDATES[int(k)])
    size = int(fleet_size)
    charge_table = VERTIPORT_CHARGE_RATE

    class StrictClosedLoopScenario(OriginalScenario):
        """同一飞机实际往返；所有新增字段仅供守恒断言和离线诊断。"""
        def __init__(self, *args, **kwargs):
            self._strict_initialized = False
            self._strict_size = size
            self._strict_ids = set()
            self._strict_log = os.environ.get("UAM_STRICT_LOG_EVENTS") == "1"
            self.strict_events = []
            self.strict_resources = []
            self.strict_stats = {}
            super().__init__(*args, **kwargs)

        def reset(self, seed=None):
            self._strict_initialized = False
            self._strict_ids = set()
            self.strict_events = []
            self.strict_resources = []
            self.strict_stats = dict(initial_fleet=size, service_arrivals=0, empty_departures=0,
                                     empty_arrivals=0, empty_flight_minutes=0., empty_energy_kwh=0.,
                                     charging_aircraft_minutes=0, idle_aircraft_minutes=0,
                                     fixed_fleet_assertions=0, initial_allocation={})
            state = super().reset(seed=seed)
            self._strict_assert()
            return state

        def _strict_event(self, event, aircraft, **extra):
            if self._strict_log:
                self.strict_events.append(dict(time=self.time, event=event, aircraft=str(aircraft.id),
                    station=str(aircraft.current_vertiport_id), state=aircraft.state.name,
                    battery_kwh=float(aircraft.battery_kwh), **extra))

        def _strict_assert(self):
            if not self._strict_initialized:
                return
            all_ids = set(self._all_evtols)
            builder_ids = set(self.evtols.evtols)
            if all_ids != self._strict_ids or builder_ids != self._strict_ids or len(all_ids) != size:
                raise AssertionError("飞机ID或数量不守恒；不会通过补机掩盖错误")
            local_ids = []
            for vid, planes in self.vertiports.evtols_at_vertiport.items():
                for plane in planes:
                    local_ids.append(plane.id)
                    if plane.state == VehicleState.FLYING or str(plane.current_vertiport_id) != str(vid):
                        raise AssertionError("飞机在站登记和飞行状态不一致")
            if len(local_ids) != len(set(local_ids)):
                raise AssertionError("同一飞机重复登记在多个机场")
            for plane in self._all_evtols.values():
                if plane.state != VehicleState.FLYING and plane.id not in local_ids:
                    raise AssertionError("非飞行飞机未登记到所在地机场")
                if not -1e-6 <= plane.battery_kwh <= plane.spec.battery_capacity_kwh + 1e-6:
                    raise AssertionError("飞机电量越界")
            self.strict_stats["fixed_fleet_assertions"] += 1

        def _maintain_evtol_capacity(self):
            if not self._strict_initialized:
                if self._all_evtols or self.evtols.evtols:
                    raise AssertionError("固定机队初始化前应为空，不叠加作者自动补机")
                weights = {v: int(self.vertiports.vertiport_evtol_capacity[v]) for v in candidates}
                total = sum(weights.values())
                raw = {v: size * weights[v] / total for v in candidates}
                allocation = {v: int(math.floor(raw[v])) for v in candidates}
                order = sorted(candidates, key=lambda v: (-(raw[v] - allocation[v]), int(v)))
                for v in order[:size - sum(allocation.values())]:
                    allocation[v] += 1
                self.strict_stats["initial_allocation"] = allocation
                index = 0
                for vid in candidates:
                    for _ in range(allocation[vid]):
                        eid = "strict_fixed_" + str(index)
                        self._spawn_landing_evtol(eid, vid)
                        self._strict_ids.add(eid)
                        index += 1
                self.eVTOL_last_id = index
                self._strict_initialized = True
            else:
                self._strict_dispatch_returns()
            self._strict_assert()

        def _strict_dispatch_returns(self):
            # 沿用responsive LQ：只看已经到站的等候队列，不偷看未来乘客。
            queue = {v: len(self.vertiports.vertiport_list[v].person_list) for v in candidates}
            available = sorted((e for e in self.vertiports.evtols_at_vertiport.get("2", [])
                                if e.state == VehicleState.IDLE and not e.passenger_ids), key=lambda e: str(e.id))
            for plane in available:
                ready = {v: sum(e.state == VehicleState.IDLE and not e.passenger_ids
                                for e in self.vertiports.evtols_at_vertiport[v]) for v in candidates}
                target = max(candidates, key=lambda v: (queue[v], -ready[v], -int(v)))
                if queue[target] <= 0:
                    break
                if self._strict_start_empty(plane, target):
                    # 仍按原飞机座位数扣一个航班负载；没有改成单客服务。
                    queue[target] = max(0, queue[target] - int(plane.spec.capacity))

        def _strict_start_empty(self, plane, target):
            if plane.state != VehicleState.IDLE or plane.passenger_ids:
                return False
            origin = str(plane.current_vertiport_id)
            points = self.vertiports.vertiport_list
            distance = self.vertiports._calculate_distance(points[origin].vertiport_position,
                                                           points[target].vertiport_position)
            energy = distance * plane.spec.energy_consumption_kwh_per_km
            if plane.battery_kwh < energy:
                plane.state = VehicleState.CHARGING
                return False
            plane.target_vertiport_id = target
            plane.passenger_ids = []
            plane.remaining_time = distance / plane.spec.max_speed * 60.
            plane.flight_distance_km = distance
            plane.planned_energy_kwh = energy
            plane.battery_kwh -= energy
            plane.state = VehicleState.FLYING
            self.vertiports.evtols_at_vertiport[origin].remove(plane)
            self.strict_stats["empty_departures"] += 1
            self.strict_stats["empty_flight_minutes"] += plane.remaining_time
            self.strict_stats["empty_energy_kwh"] += energy
            self._strict_event("EMPTY_DEPART", plane, target=target, distance=distance)
            return True

        def _handle_evtol_arrivals(self):
            events = []
            for plane in self._all_evtols.values():
                if not getattr(plane, "just_arrived", False):
                    continue
                passengers = list(plane.passenger_ids)
                vid = str(plane.current_vertiport_id)
                if passengers:
                    for pid in passengers:
                        self.persons.persons[pid].state = "finished"
                        self.persons.persons[pid].sub_state = "arrived"
                    events.append((plane.id, passengers, vid))
                    self.strict_stats["service_arrivals"] += 1
                    self._strict_event("SERVICE_ARRIVE", plane, passengers=passengers)
                else:
                    self.strict_stats["empty_arrivals"] += 1
                    self._strict_event("EMPTY_ARRIVE", plane)
                plane.passenger_ids.clear()
                plane.just_arrived = False
                local = self.vertiports.evtols_at_vertiport[vid]
                if plane not in local:
                    local.append(plane)
                # 机场拥有充电设施，飞机换站后必须读取所在站的原始速率。
                plane.spec.charge_rate_kwh_per_min = float(charge_table[vid])
                plane.state = (VehicleState.IDLE if plane.battery_kwh >= plane.spec.battery_capacity_kwh
                               else VehicleState.CHARGING)
            return events

        def step(self, action=None):
            if self._strict_initialized:
                self.strict_stats["charging_aircraft_minutes"] += sum(
                    e.state == VehicleState.CHARGING for e in self._all_evtols.values())
                self.strict_stats["idle_aircraft_minutes"] += sum(
                    e.state == VehicleState.IDLE for e in self._all_evtols.values())
            result = super().step(action)
            self._strict_assert()
            if self._strict_log and self.time % 5 == 0:
                row = dict(time=self.time)
                for vid in candidates + ("2",):
                    local = self.vertiports.evtols_at_vertiport[vid]
                    row["V" + vid + "_queue"] = len(self.vertiports.vertiport_list[vid].person_list)
                    row["V" + vid + "_ready"] = sum(e.state == VehicleState.IDLE for e in local)
                    row["V" + vid + "_charging"] = sum(e.state == VehicleState.CHARGING for e in local)
                self.strict_resources.append(row)
            return result

    StrictClosedLoopScenario.__name__ = "StrictClosedLoopScenario"
    scenario_module.Scenario = StrictClosedLoopScenario
    _STRICT_INSTALLED = signature

def activate_source(root,k):
    target=_base_activate_source(root,k)
    if os.environ.get("UAM_STRICT_MODE","E0")=="E1":
        install_closed_loop(k,int(os.environ["UAM_STRICT_FLEET"]))
    return target


METHOD_FILES = {"SPF": "methods/minimum_distance.py", "STTF": "methods/minimum_sttf.py", "QTTI": "methods/minimum_qtti.py"}

STRICT_PAYLOAD_RULES = {'methods/minimum_distance.py': "'''\nAuthor: pangay 1623253042@qq.com\nDate: 2024-01-16 20:38:03\nLastEditors: pangay 1623253042@qq.com\nLastEditTime: 2024-01-21 23:01:25\n'''\n\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nfrom gym import logger, spaces\nfrom stable_baselines3.common.env_checker import check_env\n\nfrom utilss.make_env import make_env\n\n\ndef greedy_policy(state):\n    action = [0,0,0,0]\n    persons_station = state[0:4]\n    for i in range(0,4):\n        person_station = persons_station[i,:]\n        station_0 = get_time(person_station,[0,0],state[5,1])\n        station_1 = get_time(person_station,[10,10],state[7,1])\n        if station_1 < station_0:\n            action[i] = 1\n    return action\n\ndef get_time(person_station, vertiport_stateion, wait_time):\n\n    travel_time=(\n            abs(person_station[0]-vertiport_stateion[0]) \n            + abs(person_station[1]-vertiport_stateion[1])\n            )/2 # manhattan distance\n    total_time = travel_time + wait_time\n\n    return total_time\n\ndef get_uam_time(state):\n    uam_time = state[1] + state[2]\n    return uam_time\n\nat_env_generate = make_env(log_file = 'greedy.log')\ntsc_env = at_env_generate()\nstates, _ = tsc_env.reset()\ndones = False\nrewards = 0\nwhile not dones:\n    action = greedy_policy(states)\n    print(action)\n    states, reward, truncated, dones, infos = tsc_env.step(action=action)\n    reward\ntsc_env.close()\n\nprint('平均等待时间为',reward)", 'methods/minimum_sttf.py': "'''\nAuthor: pangay 1623253042@qq.com\nDate: 2024-01-16 20:38:03\nLastEditors: pangay 1623253042@qq.com\nLastEditTime: 2024-01-21 23:01:25\n'''\n#%%\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nfrom gym import logger, spaces\nfrom stable_baselines3.common.env_checker import check_env\n\nfrom utilss.make_env import make_env\n\n#%%\ndef greedy_policy(state):\n    action = [0,0,0,0]\n    persons_station = state[0:4]\n    for i in range(0,4):\n        person_station = persons_station[i,:]\n        station_0 = get_time(person_station,[0,0],state[5,1])\n        station_1 = get_time(person_station,[10,10],state[7,1])\n        if station_1 < station_0:\n            action[i] = 1\n    return action\n\ndef get_time(person_station, vertiport_stateion, wait_time):\n\n    travel_time=(\n            abs(person_station[0]-vertiport_stateion[0]) \n            + abs(person_station[1]-vertiport_stateion[1])\n            )/1.5 # manhattan distance\n    total_time = travel_time + wait_time\n\n    return total_time\n\ndef get_uam_time(state):\n    uam_time = state[1] + state[2]\n    return uam_time\n\nat_env_generate = make_env(log_file = 'greedy.log')\ntsc_env = at_env_generate()\nstates, _ = tsc_env.reset()\nstates = states[-1,:,:]\ndones = False\nrewards = 0\nwhile not dones:\n    action = greedy_policy(states)\n    states, reward, truncated, dones, infos = tsc_env.step(action=action)\n    states = states[-1,:,:]\ntsc_env.close()\n\nprint('平均等待时间为',reward)", 'methods/minimum_qtti.py': "'''\nAuthor: pangay 1623253042@qq.com\nDate: 2024-01-16 20:38:03\nLastEditors: pangay 1623253042@qq.com\nLastEditTime: 2024-01-21 23:01:25\n'''\n#%%\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nfrom gym import logger, spaces\nfrom stable_baselines3.common.env_checker import check_env\n\nfrom utilss.make_env import make_env\n\n#%%\ndef queue_policy(state,infos):\n    action = [0,0,0,0]\n    info = info_wrapper(infos)\n    persons_station = state[0:4]\n    for i in range(0,4):\n        person_station = persons_station[i,:]\n        station_0 = get_time(person_station,[0,0],state[5,1],info[0]/(2.5*12))\n        station_1 = get_time(person_station,[10,10],state[7,1],info[1]/(1.5*4))\n        if station_1 < station_0:\n            action[i] = 1\n    return action\n\ndef get_time(person_station, vertiport_stateion, wait_time, E_lamda):\n\n    travel_time=(\n            abs(person_station[0]-vertiport_stateion[0]) \n            + abs(person_station[1]-vertiport_stateion[1])\n            )/1.5 # manhattan distance\n    E_wait_time = travel_time * E_lamda\n    print(travel_time,wait_time,E_wait_time)\n    total_time = travel_time + wait_time + E_wait_time\n\n    return total_time\n\ndef get_uam_time(state):\n    uam_time = state[1] + state[2]\n    return uam_time\n\ndef info_wrapper(info):\n    temp_state = [0,0,0,0]\n    for nam in info['persons'].person:\n        if info['persons'].person[nam].state == 'v':\n            vertiport_up_position =int(info['persons'].person[nam].vertiport_up_position)\n            temp_state[vertiport_up_position] += 1\n\n    return temp_state\nat_env_generate = make_env(log_file = 'greedy.log')\ntsc_env = at_env_generate()\nstates, _ = tsc_env.reset()\nstates = states[-1,:,:]\ndones = False\nrewards = 0\naction = [1,1,1,1]\nwhile not dones:\n    states, reward, truncated, dones, infos = tsc_env.step(action = action)\n    states = states[-1,:,:]\n    action = queue_policy(states, infos)\ntsc_env.close()\n\nprint('平均等待时间为',reward)"}

def strict_rules_snapshot():
    return {"files": STRICT_PAYLOAD_RULES}

class DiscardPrint:
    """屏蔽作者QTTI内部逐候选print；完整评分另存CSV，不改原函数。"""
    def write(self, text):
        return len(text)

    def flush(self):
        return None

def numeric_constant(node):
    """只解析作者已经写死的数字表达式，不读取环境、不执行任意表达式。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left, right = numeric_constant(node.left), numeric_constant(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("QTTI原参数不再是已知的常数乘除，请人工核对")

class AuthorFunctions:
    """直接编译作者原FunctionDef，跳过旧文件顶层失配的make_env入口。"""
    def __init__(self, method):
        self.method = method
        self.path = METHOD_FILES[method]
        self.text = strict_rules_snapshot()["files"][self.path]
        tree = ast.parse(self.text, filename=self.path)
        names = {"greedy_policy", "queue_policy", "get_time", "info_wrapper", "get_uam_time"}
        self.nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.namespace = {"__builtins__": __builtins__}
        exec(compile(ast.Module(body=self.nodes, type_ignores=[]), self.path, "exec"), self.namespace)
        self.get_time = self.namespace["get_time"]
        self.denominators = {}
        if method == "QTTI":
            policy = next(n for n in self.nodes if n.name == "queue_policy")
            for node in ast.walk(policy):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ("station_0", "station_1"):
                    self.denominators[int(target.id[-1])] = numeric_constant(node.value.args[3].right)
            if self.denominators != {0: 30.0, 1: 6.0}:
                raise RuntimeError("QTTI源参数与本轮锁定版本不一致")
        self.function_hashes = {node.name: hashlib.sha256(ast.get_source_segment(self.text, node).encode("utf-8")).hexdigest() for node in self.nodes}

    def score(self, origin, position, waiting, incoming, vid):
        with contextlib.redirect_stdout(DiscardPrint()):
            if self.method == "QTTI":
                denominator = self.denominators[0 if vid == 0 else 1]
                return float(self.get_time(origin, position, waiting, incoming / denominator))
            return float(self.get_time(origin, position, waiting))

    def original_k2_action(self, legacy_state, legacy_info):
        with contextlib.redirect_stdout(DiscardPrint()):
            if self.method == "QTTI":
                actions = self.namespace["queue_policy"](legacy_state, legacy_info)
            else:
                actions = self.namespace["greedy_policy"](legacy_state)
        if len(set(actions)) != 1:
            raise RuntimeError("旧四乘客接口复制同一焦点乘客后，作者函数返回了不一致动作")
        return int(actions[0])

    def manifest(self):
        return dict(method=self.method,source_file=self.path,original_function_hashes=self.function_hashes,
            get_time_unchanged=True,original_policy_used_for_k2=True,
            qtti_original_denominators=self.denominators,
            qtti_new_stations="仅K3/K4新增站复制V1分母6；V0=30、V1=6不动",
            spf_warning="作者minimum_distance实际含等待项，不能称论文纯距离SPF")

def adapter_decision(scenario, k, author, np):
    """新旧字段只读转换；不调用改进的排队、充电、ETA或预测模型。"""
    candidates = CANDIDATES[k]
    waiting_ids = scenario.waiting_decisions
    if not waiting_ids:
        return 0, None
    pid = waiting_ids[0]
    person = scenario.persons.persons[pid]
    origin = list(person.origin_position)
    destination = list(person.destination_position)
    # 旧代码的v表示前往机场；新状态机用enroute/to_vertiport拆成两个字段。
    incoming = {vid: 0 for vid in candidates}
    old_persons = {}
    for other_id, other in scenario.persons.persons.items():
        vid_text = str(getattr(other, "origin_vertiport_id", ""))
        enroute = getattr(other, "state", None) == "enroute" and getattr(other, "sub_state", None) == "to_vertiport"
        if enroute and vid_text in {str(v) for v in candidates}:
            vid = int(vid_text)
            incoming[vid] += 1
            # 作者旧info_wrapper只有四个槽；原两站校验只传入0/1的记录。
            if vid in (0, 1):
                old_persons[other_id] = SimpleNamespace(state="v", vertiport_up_position=vid)
    positions, wait_values, queues, scores = {}, {}, {}, []
    for vid in candidates:
        station = scenario.vertiports.vertiport_list[str(vid)]
        positions[vid] = list(station.vertiport_position)
        # 直接调用作者已有的get_wait_time：int(wait_person/volume)，不自行估算。
        wait_values[vid] = float(station.get_wait_time())
        queues[vid] = int(station.wait_person)
        score = author.score(origin, positions[vid], wait_values[vid], incoming[vid], vid)
        if not math.isfinite(score):
            raise ValueError("作者评分出现非有限值")
        scores.append(score)
    if positions[0] != [0, 0] or positions[1] != [10, 10]:
        raise RuntimeError("原0/1站坐标发生变化，不允许静默修改作者规则")
    # 原函数一次读四名乘客；复制同一已揭示焦点，不生成额外决策或乘客。
    legacy = np.zeros((9, 4), dtype=float)
    legacy[:4] = np.asarray(origin + destination, dtype=float)
    legacy[5, 1], legacy[7, 1] = wait_values[0], wait_values[1]
    legacy_info = {"persons": SimpleNamespace(person=old_persons)}
    original = author.original_k2_action(legacy, legacy_info)
    first_two = 1 if scores[1] < scores[0] else 0
    if original != first_two:
        raise RuntimeError("接口转换改变了作者两站动作，立即停止该条件")
    # 原比较使用严格小于；并列时保留原先排在前面的候选。
    action = original if k == 2 else min(range(k), key=lambda index: (scores[index], index))
    return action, dict(time=scenario.time,passenger=pid,origin=origin,destination=destination,
        chosen_airport=candidates[action],candidates=list(candidates),scores=scores,
        author_wait_time=[wait_values[v] for v in candidates],queue=[queues[v] for v in candidates],
        committed_ground_incoming=[incoming[v] for v in candidates],original_k2_action=original,
        expanded_choice=(k > 2 and action >= 2))

def ready_vector(s,k):
    """仅从已承诺乘客、飞机、电量构造共同时间参照；不看未揭示请求。"""
    waiting=s.waiting_decisions
    focal=s.persons.persons[waiting[0]] if waiting else None
    vids=CANDIDATES[k]
    access={v:float(s.vehicles.estimate_travel_time(focal.origin_position,s.vertiports.vertiport_list[str(v)].vertiport_position)) if focal else 0. for v in vids}
    common=statistics.mean(access.values())
    out=[]
    for vid in vids:
        v=str(vid)
        incoming=[max(0.,float(p.current_timer or 0)) for p in s.persons.persons.values()
                  if str(getattr(p,"origin_vertiport_id",""))==v and p.state=="enroute" and p.sub_state=="to_vertiport"]
        releases=[]
        for ac in s._all_evtols.values():
            state=ac.state.name
            if str(ac.current_vertiport_id)==v and state in ("IDLE","WAITING","CHARGING"):
                release=max(0.,ac.spec.battery_capacity_kwh-ac.battery_kwh)/max(ac.spec.charge_rate_kwh_per_min,1e-9)
                releases.append(release)
            elif state=="FLYING" and str(ac.target_vertiport_id)==v:
                from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
                charge=max(0.,ac.spec.battery_capacity_kwh-ac.battery_kwh)/max(float(VERTIPORT_CHARGE_RATE[v]),1e-9)
                releases.append(max(0.,float(ac.remaining_time))+charge)
        q=len(s.vertiports.vertiport_list[v].person_list)
        out.extend([common/60.,access[vid]/60.,q/300.,sum(t<=common for t in incoming)/300.,
                    sum(t<=common for t in releases)/12.,sum(t<=common+15 for t in releases)/12.,
                    min(releases,default=120.)/120.,len(releases)/12.])
    return out


def configure_case(root,task):
    os.environ["UAM_STRICT_MODE"]=task["environment"]
    os.environ["UAM_STRICT_FLEET"]=str(task["fleet"])
    os.environ["UAM_STRICT_LOG_EVENTS"]="1"
    materialize_source(root)
    install_learning_classes()
    activate_source(root,task["k"])
    from loguru import logger
    logger.remove()
    logging.disable(logging.CRITICAL)
    install_diagnostic_classes()


def load_source_norm(path):
    """对自己旧权重配对的类名做显式映射，避免pickle导入其他矩阵入口。"""
    class SourceNormLoader(pickle.Unpickler):
        def find_class(self,module,name):
            if name=="SourceVecNormalize" and (module in ("__main__","__mp_main__") or module.startswith("train_uam_")):
                return SourceVecNormalize
            return super().find_class(module,name)
    with open(path,"rb") as f:
        return SourceNormLoader(f).load()


def install_diagnostic_classes():
    global DiagnosticEnv,DiagnosticFactory,ResidualPolicy,DiagnosticPPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.distributions import CategoricalDistribution
    from torch import nn

    class DiagnosticEnv(gym.Env):
        """冻结源归一化，新增物理时间自行缩放；不重新拟合源obs统计。"""
        def __init__(self,root,task,trace,horizon):
            self.raw=SourceEnv(root,task["k"],horizon,trace)
            self.task=task
            # 只读取本项目自己生成并在索引中校验过的模型归一化；绝不读取陌生归档pickle。
            self.norm=load_source_norm(task["source_vecnorm"])
            self.norm.training=False
            self.norm.norm_reward=False
            self.action_space=self.raw.action_space
            size=self.raw.observation_space.shape[0]+task["k"]*8+1
            self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(size,),dtype=np.float32)
        def transform(self,obs):
            return np.r_[self.norm.normalize_obs(obs.copy()),ready_vector(self.raw.scenario,self.task["k"]),
                         float(bool(self.raw.scenario.waiting_decisions))].astype(np.float32)
        def reset(self,*,seed=None,options=None):
            x,info=self.raw.reset(seed=seed,options=options)
            return self.transform(x),info
        def step(self,action):
            x,r,t,tr,info=self.raw.step(int(action))
            return self.transform(x),r,t,tr,info
        def close(self):
            self.raw.close()

    class DiagnosticFactory:
        def __init__(self,root,task,trace,horizon,rank,directory):
            self.values=str(root),task,str(trace),horizon,rank,str(directory)
        def __call__(self):
            root,task,trace,horizon,rank,directory=self.values
            from stable_baselines3.common.utils import set_random_seed
            set_random_seed(task["seed"]+rank)
            work=Path(directory)/("env_"+str(rank))
            work.mkdir(parents=True,exist_ok=True)
            os.chdir(work)
            return Monitor(DiagnosticEnv(root,task,trace,horizon),str(work/"monitor.csv"))

    class ResidualPolicy(ActorCriticPolicy):
        """logπ=冻结源logπ＋小残差；残差头为零时逐状态等于源策略。"""
        def __init__(self,*args,source_model,k,shared_on=False,**kwargs):
            self.source_model=source_model
            self.k=int(k)
            self.shared_on=bool(shared_on)
            self.frame=4+8*self.k
            self.temporal_dim=N_FRAMES*self.frame
            super().__init__(*args,**kwargs)
        def _build(self,lr_schedule):
            self.frozen=SourcePPO.load(self.source_model,device=self.device).policy
            for p in self.frozen.parameters():
                p.requires_grad_(False)
            self.frozen.eval()
            self.source_dim=self.frozen.observation_space.shape[0]
            self.residual_encoder=nn.Sequential(nn.Linear(self.frame+8*self.k,64),nn.LayerNorm(64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
            self.residual_actor=nn.Linear(64,self.k)
            self.residual_value=nn.Linear(64,1)
            self.inverse=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,self.k))
            for head in (self.residual_actor,self.residual_value):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.action_dist=CategoricalDistribution(self.k)
            self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
        def encode(self,obs):
            current=obs[:,self.temporal_dim-self.frame:self.temporal_dim]
            shared=obs[:,self.source_dim:self.source_dim+8*self.k]
            if not self.shared_on:
                shared=torch.zeros_like(shared)
            return self.residual_encoder(torch.cat([current,shared],-1))
        def outputs(self,obs):
            self.frozen.eval()
            with torch.no_grad():
                base=self.frozen.get_distribution(obs[:,:self.source_dim]).distribution.logits
                value=self.frozen.predict_values(obs[:,:self.source_dim])
            z=self.encode(obs)
            return base+self.residual_actor(z),value+self.residual_value(z)
        def forward(self,obs,deterministic=False):
            logits,value=self.outputs(obs)
            d=self.action_dist.proba_distribution(action_logits=logits)
            a=d.get_actions(deterministic=deterministic)
            return a,value,d.log_prob(a)
        def get_distribution(self,obs):
            logits,_=self.outputs(obs)
            return self.action_dist.proba_distribution(action_logits=logits)
        def predict_values(self,obs):
            return self.outputs(obs)[1]
        def evaluate_actions(self,obs,actions):
            logits,value=self.outputs(obs)
            d=self.action_dist.proba_distribution(action_logits=logits)
            return value,d.log_prob(actions),d.entropy()
        def set_training_mode(self,mode):
            super().set_training_mode(mode)
            self.frozen.eval()
        def _get_constructor_parameters(self):
            d=super()._get_constructor_parameters()
            d.update(source_model=self.source_model,k=self.k,shared_on=self.shared_on)
            return d

    class DiagnosticPPO(PPO):
        def __init__(self,*args,aux_on=False,**kwargs):
            self.aux_on=aux_on
            super().__init__(*args,**kwargs)
        def train(self):
            b=self.rollout_buffer
            raw=b.observations.copy()
            actions=b.actions.copy()
            starts=b.episode_starts.copy()
            probe=torch.as_tensor(raw.reshape(-1,raw.shape[-1])[:128],device=self.device)
            with torch.no_grad():
                before=self.policy.get_distribution(probe).distribution.probs.detach().clone()
            super().train()
            with torch.no_grad():
                ppo_after=self.policy.get_distribution(probe).distribution.probs.detach().clone()
            loss_value=None
            if self.aux_on:
                # 相邻一分钟逆模型作为被检验因素，不预称它具有严格因果信用。
                valid=(raw[:-1,:,-1]>.5)&(starts[1:]<.5)
                ix=np.flatnonzero(valid.reshape(-1))
                if len(ix):
                    ix=np.random.choice(ix,min(512,len(ix)),replace=False)
                    x=torch.as_tensor(raw[:-1].reshape(-1,raw.shape[-1])[ix],device=self.device)
                    y=torch.as_tensor(raw[1:].reshape(-1,raw.shape[-1])[ix],device=self.device)
                    a=torch.as_tensor(actions[:-1].reshape(-1,actions.shape[-1])[ix,0].astype(np.int64),device=self.device)
                    logits=self.policy.inverse(torch.cat([self.policy.encode(x),self.policy.encode(y)],-1))
                    loss=.1*torch.nn.functional.cross_entropy(logits,a)
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad],.5)
                    self.policy.optimizer.step()
                    loss_value=float(loss.detach().cpu())
            with torch.no_grad():
                aux_after=self.policy.get_distribution(probe).distribution.probs.detach()
            def kl(a,b):
                return float((a*(a.clamp_min(1e-10).log()-b.clamp_min(1e-10).log())).sum(-1).mean().cpu())
            self.logger.record("diag/ppo_policy_KL",kl(before,ppo_after))
            self.logger.record("diag/aux_policy_KL",kl(ppo_after,aux_after))
            self.logger.record("diag/total_policy_KL",kl(before,aux_after))
            self.logger.record("diag/aux_loss",loss_value or 0.)
            self.logger.record("diag/aux_on",int(self.aux_on))
            self.logger.record("diag/shared_on",int(self.policy.shared_on))


class SafeDiagnosticFactory:
    """spawn只传基础配置；子进程重新初始化，不序列化父进程动态注册类。"""
    def __init__(self,root,task,trace,horizon,rank,directory):
        self.values=str(root),task,str(trace),horizon,rank,str(directory)
    def __call__(self):
        root,task,trace,horizon,rank,directory=self.values
        work=Path(directory)/("env_"+str(rank))
        work.mkdir(parents=True,exist_ok=True)
        os.chdir(work)
        configure_case(root,task)
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.monitor import Monitor
        set_random_seed(task["seed"]+rank)
        return Monitor(DiagnosticEnv(root,task,trace,horizon),str(work/"monitor.csv"))


def source_selection(source_root,k,seed,metric):
    matches=[]
    for p in Path(source_root).glob(f"cells/*K{k}*UAGMC_SOURCE_L*s{seed}/curve.csv"):
        with p.open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                key="ATT_completed_only" if metric=="ATT_CO" else "ATT"
                try:
                    value=float(r[key])
                    step=int(r["step"])
                except (ValueError,KeyError,TypeError):
                    continue
                if not math.isfinite(value):
                    continue
                marker=p.parent/"checkpoints"/f"step_{step:09d}.json"
                rec=read_json(marker)
                if not rec:
                    continue
                model=marker.parent/Path(rec["model"]).name
                norm=marker.parent/Path(rec["vecnorm"]).name
                if model.is_file() and norm.is_file():
                    matches.append((value,step,str(model.resolve()),str(norm.resolve()),float(r.get("completion",0))))
    if not matches:
        raise FileNotFoundError(f"未找到K{k} seed{seed}的可配对源Best checkpoint；不允许自动从随机初始化替代")
    value,step,model,norm,completion=min(matches)
    return dict(source_model=model,source_vecnorm=norm,source_best_ATT=value,source_step=step,
                source_best_completion=completion,source_model_sha256=digest(model),source_norm_sha256=digest(norm))


def replay_diagnostic(model,root,task,trace,horizon,seed):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    env=DiagnosticEnv(root,task,trace,horizon)
    obs,_=env.reset(seed=seed)
    counts=[0]*task["k"]
    area=0.
    while True:
        action,_=model.predict(obs,deterministic=True)
        if env.raw.scenario.waiting_decisions:
            counts[int(action)]+=1
        s=env.raw.scenario
        area+=sum(pid not in set(s.finished_ids) for pid in s.persons.persons)
        obs,_,t,tr,_=env.step(int(action))
        if t or tr:
            break
    result=episode_metrics(env.raw,counts,area)
    s=env.raw.scenario
    result.update(mode="DET",eval_seed=seed,strict_environment=task["environment"],fleet_size=task["fleet"],
                  lifecycle=getattr(s,"strict_stats",{}),resource_timeline=getattr(s,"strict_resources",[]),
                  aircraft_events=getattr(s,"strict_events",[]))
    env.close()
    return result


def probe_worker(cli,task):
    root=Path(cli.output_root)
    directory=root/"probes"/task["id"]
    directory.mkdir(parents=True,exist_ok=True)
    os.chdir(directory)
    configure_case(root,task)
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(123)
    model=SourcePPO.load(task["source_model"],device="cpu")
    norm=load_source_norm(task["source_vecnorm"])
    norm.training=False
    norm.norm_reward=False
    env=SourceEnv(root,task["k"],cli.horizon,cli.trace)
    obs,_=env.reset(seed=123)
    probes=[]
    every=15
    decisions=0
    fields=[]
    while True:
        waiting=env.scenario.waiting_decisions
        if waiting:
            decisions+=1
            if decisions%every==0 and len(probes)<cli.probe_states:
                s=env.scenario
                raw=np.asarray(env.raw.encoder.encode(s,env.raw.state))
                # 原8维站块中的时间槽逐通道记录零值率，不假定运行时已经发生。
                fields.append(raw.tolist())
                rs,ns=random.getstate(),np.random.get_state()
                costs=[]
                for action in range(task["k"]):
                    branch=copy.deepcopy(env)
                    random.setstate(rs)
                    np.random.set_state(ns)
                    x,_,_,_,_=branch.step(action)
                    area=0.
                    for _ in range(cli.branch_minutes):
                        sc=branch.scenario
                        area+=sum(pid not in set(sc.finished_ids) for pid in sc.persons.persons)
                        a,_=model.predict(norm.normalize_obs(x.copy()),deterministic=True)
                        x,_,t,tr,_=branch.step(int(a))
                        if t or tr:
                            break
                    costs.append(area)
                    branch.close()
                random.setstate(rs)
                np.random.set_state(ns)
                chosen,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
                probes.append(dict(time=s.time,passenger=waiting[0],chosen=int(chosen),
                    branch_costs=costs,chosen_cost_excess=costs[int(chosen)]-min(costs),
                    ready_features=ready_vector(s,task["k"]),queues=[len(s.vertiports.vertiport_list[str(v)].person_list) for v in CANDIDATES[task["k"]]],
                    boundary="共同随机起点、同一后续源策略的有限时域人时；非全局最优Q"))
        action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
        obs,_,t,tr,_=env.step(int(action))
        if t or tr:
            break
    zeros=(np.asarray(fields)==0).mean(0).tolist() if fields else []
    save_json(directory/"PROBE.json",dict(task=task,probes=probes,raw_channel_zero_fraction=zeros,
        sampled_states=len(probes),new_training_steps=0,probe_seed=123))
    env.close()


def train_diagnostic(cli,task):
    root=Path(cli.output_root)
    directory=root/"cells"/task["id"]
    directory.mkdir(parents=True,exist_ok=True)
    os.chdir(directory)
    configure_case(root,task)
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback
    set_random_seed(task["seed"])
    fs=[SafeDiagnosticFactory(root,task,cli.trace,cli.horizon,i,directory) for i in range(cli.n_envs)]
    vector=SubprocVecEnv(fs,start_method="spawn") if cli.n_envs>1 else DummyVecEnv(fs)
    vec=VecNormalize(vector,norm_obs=False,norm_reward=True,gamma=.99,clip_reward=10.)
    old_norm=load_source_norm(task["source_vecnorm"])
    vec.ret_rms=copy.deepcopy(old_norm.ret_rms)
    steps=math.ceil(cli.timesteps/(cli.n_envs*cli.n_steps))*cli.n_envs*cli.n_steps
    pointer=read_json(directory/"RESUME_POINTER.json") if cli.resume else None
    shared,aux=VARIANTS[task["variant"]]
    if pointer:
        vec.close()
        vector=SubprocVecEnv(fs,start_method="spawn") if cli.n_envs>1 else DummyVecEnv(fs)
        vec=VecNormalize.load(pointer["vecnorm"],vector)
        model=DiagnosticPPO.load(pointer["model"],env=vec,device=cli.device)
    else:
        model=DiagnosticPPO(ResidualPolicy,vec,aux_on=aux,learning_rate=lambda x:5e-4*x,
            policy_kwargs=dict(source_model=task["source_model"],k=task["k"],shared_on=shared,net_arch=[]),
            seed=task["seed"],device=cli.device,verbose=1,
            **{k:v for k,v in SOURCE_PPO.items() if k not in ("n_envs","n_steps")},n_steps=cli.n_steps)
        # 零残差的保真断言是正式训练内嵌检查，不是额外训练阶段。
        obs=vec.reset()
        x=torch.as_tensor(obs,device=model.device)
        with torch.no_grad():
            p=model.policy.get_distribution(x).distribution.probs
            q=model.policy.frozen.get_distribution(x[:,:model.policy.source_dim]).distribution.probs
            error=float((p-q).abs().max().cpu())
        if error>1e-6:
            raise AssertionError("零残差未逐状态复现源策略")
        save_json(directory/"FUNCTION_PRESERVATION.json",dict(max_probability_error=error,source=task["source_model"]))
    model.set_logger(configure(str(directory/"training"),["stdout","csv","tensorboard"]))
    class SaveCheckpoint(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next=((model.num_timesteps//50000)+1)*50000
        def _on_step(self):
            if self.num_timesteps>=self.next:
                item=save_pair(self.model,directory/"checkpoints",self.num_timesteps)
                save_json(directory/"RESUME_POINTER.json",item)
                self.next+=50000
            return True
    try:
        if steps>model.num_timesteps:
            model.learn(total_timesteps=steps-model.num_timesteps,reset_num_timesteps=False,callback=SaveCheckpoint())
        item=save_pair(model,directory/"checkpoints",model.num_timesteps)
        save_json(directory/"TRAIN_END.json",dict(task=task,actual_steps=model.num_timesteps,final=item))
    finally:
        vec.close()


def eval_diagnostic(cli,task):
    root=Path(cli.output_root)
    directory=root/"cells"/task["id"]
    os.chdir(directory)
    configure_case(root,task)
    rows=[]
    markers=sorted((directory/"checkpoints").glob("step_*.json"))
    if not markers:
        raise FileNotFoundError("没有完整检查点，不将未训练格伪装为评估成功")
    for marker in markers:
        d=read_json(marker)
        output=directory/"evaluation"/marker.name
        if cli.resume and output.is_file():
            rows.append(read_json(output)["summary"])
            continue
        path=marker.parent/Path(d["model"]).name
        model=DiagnosticPPO.load(path,device="cpu")
        episodes=[replay_diagnostic(model,root,task,cli.trace,cli.horizon,seed) for seed in (123,124)]
        summary=dict(step=d["step"],ATT=mean(e["ATT"] for e in episodes) if all(e["full"] for e in episodes) else None,
                     ATT_completed_only=mean(e["ATT_completed_only"] for e in episodes),completion=mean(e["completion"] for e in episodes))
        save_json(output,dict(task=task,summary=summary,episodes=episodes))
        rows.append(summary)
    save_csv(directory/"curve.csv",rows)
    field="ATT_completed_only" if cli.metric=="ATT_CO" else "ATT"
    valid=[r for r in rows if r.get(field) is not None]
    best=min(valid,key=lambda r:r[field]) if valid else None
    save_json(directory/"RESULT.json",dict(task=task,primary=cli.metric,Best=best,source_Best=task["source_best_ATT"],
        plateau_reference=mean(r.get(field) for r in rows if r["step"]>=.8*max(x["step"] for x in rows)) if rows else None,
        scope="同源Best锚点的残差续训，不与随机初始化总训练量直接排名"))


def rules_diagnostic(cli,task):
    root=Path(cli.output_root)
    directory=root/"rules"/(task["environment"]+"__K"+str(task["k"]))
    directory.mkdir(parents=True,exist_ok=True)
    os.chdir(directory)
    configure_case(root,task)
    episodes=[]
    for rule in ("SPF","STTF","QTTI"):
        author=AuthorFunctions(rule)
        for seed in (123,124):
            random.seed(seed)
            np.random.seed(seed)
            env=SourceEnv(root,task["k"],cli.horizon,cli.trace)
            env.reset(seed=seed)
            counts=[0]*task["k"]
            area=0.
            while True:
                a,detail=adapter_decision(env.scenario,task["k"],author,np)
                if detail is not None:
                    counts[a]+=1
                s=env.scenario
                area+=sum(pid not in set(s.finished_ids) for pid in s.persons.persons)
                _,_,t,tr,_=env.step(a)
                if t or tr:
                    break
            result=episode_metrics(env,counts,area)
            result.update(mode=rule,rule=rule,eval_seed=seed,step=0,environment=task["environment"],k=task["k"],
                          fleet=task["fleet"] if task["environment"]=="E1" else None)
            episodes.append(result)
            env.close()
        save_json(directory/(rule+"_AUTHOR_FUNCTIONS.json"),author.manifest())
    save_json(directory/"RULES.json",dict(task=task,episodes=episodes))


def pack_diagnostic(root):
    root=Path(root)
    package=root.with_name(root.name+"_DATAONLY.tar.gz")
    with tarfile.open(package,"w:gz") as arc:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".json",".csv",".txt",".md",".log") and "_author_source" not in p.parts:
                arc.add(p,arcname=root.name+"/"+p.relative_to(root).as_posix())
    print("[数据包]",package,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-e0",default="serial_runs/uam_source_K234_5methods_4seeds_60m_v1")
    p.add_argument("--source-e1",default="serial_runs/uam_strictE1_LQ_fleet40m_methods60m_v1")
    p.add_argument("--output-root",default="serial_runs/uam_E01_share_ac_diagnosis_48m_v1")
    p.add_argument("--timesteps",type=int,default=1000000)
    p.add_argument("--n-envs",type=int,default=5)
    p.add_argument("--n-steps",type=int,default=4096)
    p.add_argument("--device",default="cuda")
    p.add_argument("--horizon",type=int,default=600)
    p.add_argument("--fleet",type=int,default=12)
    p.add_argument("--metric",choices=("ATT_CO","ATT"),default="ATT_CO")
    p.add_argument("--trace",default=None)
    p.add_argument("--probe-states",type=int,default=12)
    p.add_argument("--branch-minutes",type=int,default=60)
    p.add_argument("--phase",choices=("all","probe","rules","train","eval","pack"),default="all")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--worker",default=None,help=argparse.SUPPRESS)
    p.add_argument("--worker-phase",default=None,help=argparse.SUPPRESS)
    cli=p.parse_args()
    for key in ("source_e0","source_e1","output_root"):
        setattr(cli,key,str(Path(getattr(cli,key)).resolve()))
    root=Path(cli.output_root)
    root.mkdir(parents=True,exist_ok=True)
    materialize_source(root)
    cli.trace=str(Path(cli.trace).resolve()) if cli.trace else str(root/"_author_source"/source_snapshot()["commit"]/"train_data/passengers_300.csv")
    if cli.worker:
        task=json.loads(cli.worker)
        directory=root/("probes" if cli.worker_phase=="probe" else "cells")/task["id"]
        if cli.worker_phase=="rules":
            directory=root/"rules"/(task["environment"]+"__K"+str(task["k"]))
        directory.mkdir(parents=True,exist_ok=True)
        {"probe":probe_worker,"rules":rules_diagnostic,"train":train_diagnostic,"eval":eval_diagnostic}[cli.worker_phase](cli,task)
        return
    if cli.phase=="pack":
        pack_diagnostic(root)
        return
    if min(cli.timesteps,cli.n_envs,cli.n_steps,cli.horizon,cli.probe_states,cli.branch_minutes)<=0:
        p.error("步数、并行和回放预算必须为正")
    tasks=[]
    errors=[]
    for env in ("E0","E1"):
        for k in (2,3,4):
            for source_seed,new_seed in ((601,801),(602,802)):
                try:
                    source=source_selection(cli.source_e0 if env=="E0" else cli.source_e1,k,source_seed,cli.metric)
                except FileNotFoundError as ex:
                    errors.append(str(ex))
                    continue
                for variant in VARIANTS:
                    tasks.append(dict(id=f"{env}__K{k}__{variant}__s{new_seed}",environment=env,k=k,fleet=cli.fleet if env=="E1" else None,
                        seed=new_seed,source_seed=source_seed,variant=variant,**source))
    contract=dict(version=VERSION,timesteps=cli.timesteps,n_envs=cli.n_envs,n_steps=cli.n_steps,horizon=cli.horizon,fleet=cli.fleet,
        metric=cli.metric,trace_hash=digest(cli.trace),tasks=tasks)
    old=read_json(root/"PLAN.json")
    if old is not None and old!=contract:
        raise RuntimeError("目录冻结计划不同，不把不同源checkpoint或预算混入旧目录")
    save_json(root/"PLAN.json",contract)
    save_json(root/"MISSING_SOURCES.json",errors)
    if not tasks:
        raise RuntimeError("没有任何可用源模型；见MISSING_SOURCES.json。不允许悄悄改为从零训练")
    done=failed=0
    phases=("probe","rules","train","eval") if cli.phase=="all" else (cli.phase,)
    for phase in phases:
        selected=tasks if phase not in ("probe","rules") else [t for t in tasks if t["variant"]=="R00_BASE_RESIDUAL" and (phase!="rules" or t["source_seed"]==601)]
        for i,task in enumerate(selected):
            folder=root/("probes" if phase=="probe" else "cells")/task["id"]
            if phase=="rules":
                folder=root/"rules"/(task["environment"]+"__K"+str(task["k"]))
            folder.mkdir(parents=True,exist_ok=True)
            marker=folder/({"probe":"PROBE.json","rules":"RULES.json","train":"TRAIN_END.json","eval":"RESULT.json"}[phase])
            if cli.resume and marker.is_file():
                done+=1
                continue
            print(f"[{phase} {i+1}/{len(selected)}] {task['id']}",flush=True)
            cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker",json.dumps(task),"--worker-phase",phase,
                 "--output-root",str(root),"--trace",cli.trace,"--device",cli.device,"--timesteps",str(cli.timesteps),
                 "--n-envs",str(cli.n_envs),"--n-steps",str(cli.n_steps),"--horizon",str(cli.horizon),"--metric",cli.metric,
                 "--probe-states",str(cli.probe_states),"--branch-minutes",str(cli.branch_minutes)]
            if cli.resume:
                cmd.append("--resume")
            childenv=dict(os.environ,PYTHONUTF8="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            with (folder/(phase+".log")).open("a",encoding="utf-8") as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=childenv)
            if result.returncode:
                failed+=1
                save_json(folder/(phase+"_FAILED.json"),dict(code=result.returncode,task=task))
            else:
                done+=1
            save_json(root/"progress.json",dict(phase=phase,done_operations=done,failed_operations=failed,
                planned_train_runs=len(tasks),actual_planned_steps=len(tasks)*math.ceil(cli.timesteps/(cli.n_envs*cli.n_steps))*cli.n_envs*cli.n_steps))
    pack_diagnostic(root)


if __name__=="__main__":
    main()

