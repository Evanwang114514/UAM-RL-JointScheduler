#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""作者源环境：K2/K3/K4 × 五种表示 × 四个训练种子。

这是单文件正式实验入口，不修改调用项目的任何环境文件。
作者源码按提交锁定并内嵌，运行时释放到本轮结果目录的隔离副本。
只将训练入口的LSTM注册和帧形状接通；不修正作者仿真中的行为。
源码已有自动补机、充电、满座服务、奖励、结束条件均原样保留。
完整rollout下每run实际1,003,520步，60run实际60,211,200步。

默认：先串行完成所有训练，再统一回放；可用--resume接着运行。
单cell失败后记录并继续；--retry-failed才重新尝试已失败的cell。
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
import zlib

# 仅限制本进程和自己创建的子进程线程，不改系统或他人配置。
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SOURCE_PAYLOAD = "c-ri}dv_bhkvRHQCgjNykU<Cl35vuqa#oRKulDNolVs=YA$hPMFeDLy0D=c4Myq>h%X-;*Igw>cmLGB~TXC#M;@Fl*J-)&H5V;(X{L8<OaI2~x)7{fE0A(kez31nIBB!Ujr@Okks=BJW>f}JvZPZ(pcD=bYkQ<n9w;Qe8_U&_(_Izh1S*kB=KiVwL&Q?mpJFAWP;`Y~ezPf+cKw<#iEL7UiVtiyIy~8bL(zEH2Y-X&K&bkxXbapgboSm4NP0wUgshNpldIZ|cR;q4mAa`=$@SdH!_wPwAl%f5gLtW;a*P1iMnzOUg9DcP~@6^i9{`yR%T4^siyXy63xl${(EA`qG{O{k~w%e@Fxiz=mX*sXE&32_xZ?>HyZq+SeCFfwHU0JC7mRdNo^`>*MHoV(iC_=CB0q1D(L<MR$7b_*Vm2-AFyIr^89Cj-;s9SOu+*;e&=N6l_N^Q>B*=RKD#nL=}d2rDA-=F;cqH}N&8@tD`jE3((9RQ=-a$pXXTHCEwD|6UQ*)3ICFpA-Y;xX)Pwpm<okJp>Woc4UN?NnXpsW=Di7CKcJbG=-wI`x?rbWekar*g4fba3QNZG2TRzINTIv+26dqJ%|D!lF`h;PYiR`yzn1jk9m@*(aT&^R6<gV!7OOTR0I|;YQUxk#NdOwc<jhl%PS(Yp`DsOXSuTE6sWhN7iy?mYiSL0SMJ4=%m)Fl-(u{v{8h9YjbYXVS{NUoH@4AZH8i4yIjFR!z*Pf?PA+)vCkKwD}c$OJ6|bPU8huR6lnw%m<cY%%jL?VQv}%8@>2sP7>(PU8u-Oj&2e6?EX+BrW(mr19PQTjPH}FbRFG3oHfnQI1J3bExjhe+c4U7}EnnUaUH(EY{>^3Q|8kd{SKMN|1AxNvVQ1U6{kVF=N9r9|9_I)S7|^K!6<`RoklVK9oV_rdxh77=Ik-CjgjU0K%jUd0eBSQsM-T2Zr;8sx>MlTK00|$Fv*!drvm`#b2iCB@)N=M6Il3P)1$v-bZPBdkHr!g-t(BllL=}B~3G{#~zv(~?`OQOz4i3*0TW;AoRIgS_OWJnBufB%(bEsKAQCWabSSrz;Y}RKwtu{>$zB2>MDR7#9XK$_4BxHw94!g&T%`$A%7OX9P<7lyT49E$9t=Me2usP^GY(wa10amd+@4)xm;a1yi3?HxfRa*1K2JJ)`!i?L7=_3ABXfL^lh839WQc|w?m!CSX$nCMSS(>l3VP^plv4*~R2-{n*1M0J|y_>G<EEInYpMh=NfwAVWMLG#HV;HihR;rirv%jp?OUG!rummQhuY!dh!0xI$Rp<mL!7Rr`7;*+XH!uOd@AuD&S|RH{+pI6})o$Z9b7?qbz5}8E5$vC@=K;5klg4geg@H1>W|-G<Vm=zfZi6Kw$`E%tEu1eP5W&~qQ?Ipv2(kh!x$@2rkFNY}`O@ERya|6eN5%RaY2#<+Ds1B&ov*ZT#eWU!fxj0k2s;LNxuh-GB;mL~#`zG+mDyRhiQr>6-qdz0ki}oR)%tO#h<gRr7&<OBD-AX|#MllF7?uIL@9;SzLeUreZ&Mpu_HEmC0msCauyD;zEw^pkRP8_i^FJF)?Rn&C?6=dbCL2riVyXrlv)C@wXMVlCP=Nm0g#x?-j_H@Xi|u-~$6NZBy`tqs^eI(~MB*z8v_i5d;mlUtYI&+gHQP(HPk5z+4TgXGCHqE__0Gw0fq{}Yc%2mzs@odTzXhOrCDDS2+z}{Xxbd%YTvIjr#VX{GnmbOVoWr$c57<jiZ;!=;j`#N6?(!S!Z+{2>@~SwVoHjZquRSLB?%ubDRW5a!h$97w%?13d0L3JHe;OJesMlOpzg=w3x&2iaBi#7*t?tTIb%ZlTBuYz#W5?%n&TO>~VnUuRObiz~|4Omia^s@C<f?@du#t8SNdweRi7)u?D`k*%sz9G)cV&0B2#l*ROJrgxk1yl?(7Nw`xB32G+0cG?a(?~V-L;j=Ypc)KZ+-H^lMAdC5NL+%1*EqAcvHy~!T;&+y$hFE7v6<Gu06fMI)(1M^Jh1I=XX?dPqYig#bTw3tRSkljq_ivU%9uw`upxjpRnG<kfBN*|8u(6A9|mCj30S^X7lzce1ytbtyvHc=G%obOtuJIt#E8X`w$0PwOFgU<pNT+Ib$tkrC$U8DioW|%AyPFJX5b%;VmtX@*XH`vDvm)5KG-Cwf*kX$rRfFSOb>GDHK4oZ1Y_n1IcbS&Nl%eG(0C2oPi#w2=|06N6DcCf*Mxvn~H_b+y4`<+46P&u-yIRPG3!M(3O_InrzI@@Gl4i%GjGOaH<*~FC!i`=VV|rCiMOJtDSp~?mh78DHTSgVy#dtmE1<VAThp0-%U8R&H`-<+X%!Es&0*HC$;?&cV5EQ&dVxH1uWIi&lg*TR)<RuZH_hk^Fc(jgW@-xN)woEUTxrPQ$Eab5v7dI0udEl&O!smdTc@M3B?7sCmv{hFH~v;-muU_9%x@A70-tKG5;bjzN!KRW&tFyG!H}(2!tJRTQ&l`>)&=Beic^OsH@W&+=xUJ&YW4l_N{m8u7jeUv-arT`d=S-Z-463u^#BnC>o<A-OG13o`2XklH(V}D!bKokp@-(Sc-)<r>jy7ofPO*1kniAQAmbZVRccq_r3YGrmQ#RNs?4#wX@n<8H@GGzW;!wQnysddJX3dAn8rSSBzG$^-@OX@&kogsMwk>Fp3_FtAlo{(Zd!v#`+EnMFq?+h$e>(q7^Mdi2np@^3Hs-dF}q%^ZVZCmwtG1bMu40bnkukx8+;izuX2)?ylbU9{$<E=6_rM$WhR(pE+T(T;x<<-ttWRy{NY`w>e(}9lsL(BS_GuU<C4zS|LCtm6sMNhP_mG7}fzXQmYFh?bN_YnqH-R%2~h@=X?>Uqt7K!BBwcU$>3*G1A#t!06pPM4eX&KI(=Zqr4=+gw)Bf^$KW`Tyhg5-MbbphVFKA`rF{zOBOHzIt6v&)HvjPP#_fw?GMgezi9KzwFO?iE6i*aL$pMUKOv+H<)ZOS=AK1p}r<>=lY<&5acm8)qM?v{cACw7`fTRc2V>+_`NRDg(wR_<Yz_9?F8{d8p0QN3@10@<zgS<8$HrN1WSa&fa(xR>+TIv+^N3&?#fl06_3(EG1*8Tx2dXf7;s#G$iew4FfMP?Mc!X^x&v~lH&wMWbAUtR+8?VbOwd;M|uoA=bk4Yo^`P_Gq>JA-RaZh8-%Y@GY6?YnD_KiT-~Pu)jn*8jK?Ej1i%cHDq?7VOa%fYg;1D@3c>kL)tIRT*Hg53aIr=Yic4dj_58HA@rV-#E8%|Jmk^OB+|8ZY;kKg^U>av+RdM^6%P5x@F|G^wA4}jq1kfogLgQ^(+{Ggyd6z$wTk_!;NRp`iH%~b$bEdCS#id_SWSs^RxQ-6_79%D1OO6Q;iVQA%jjXuB5<FCt-dp<iBW{Id1kMW~vVF7rl;KIonuzNsxUWVc=lC@!+%W<vZS$KlFWgK|EzPh5+5?H07ir@(lpe59Ay!O(o2?G!I#Uo&t3Odra21?Y=;Bb9>3wmVFm!tR8PfJEbg8Um>NUx-MIq4fM?;8yUs2(J58Ot7vNrGo4Dc>^7O!mMce_#ag+(U{%fTuE5}(eZaL6&d%CWB2rgN|7sf7Eh}fksyE#^7(;VOG$Dz&aOBXQT}Sjra+UnQv~Rt>=kU?JhYlV-S|F;ir*L@ZQQ8cO8u*tCMjUBDv>p)RH5bH-wIZxjl9e5HTMabM2<sVdHjL(@<D{#nRxMH7X_Sj?vJ)`lCpUpOF(!8;9HG@nIF)i$Xk&-D6@u9vaA!GWvR&|vUaiu+Tkce$hCXmr+ove{6nh{$`AB6lE;nhEg-SUfx6z_N6GRWJnP7m!`X^7Bau+A|MzQvBc&|SL8Pp5hYRh#drv_3}19&1zPYq0~GeZ?oP{w2s*gSV>{po$S3hH9^8X=pc5V15kmYOcI1FFG+$?#O1J)PcqdeYRXW(R-7>(@T%p1<l{dawKOclL3!ufbl$flm$m`@vuSN8!k*S_7@9g!p-NvrPI8r~SGrivvt5wC5`=pWY$+u<e>jb&g7J{Ht<T8wZ|BIWJ&kP`QOVu;IKIc#<B?D#cm{O$wc++f-^QrA7n6CjH8yB+xTILaN>JBBLtM%Q-w2_%%g|QPIY${YJL8=AcZ~sWLe8mC-cv{OETRS<Qr{T@r$|jz*x6(`txJtUVfZz^SyHTD?vChmi!vyFf^*S4!;6p;MN4K=L5G>>DQ$NO}CngoS<ZwqnclNw<aU;qOL5{AWf?IP3Brf7kuyu6On#NcIlrbKy+^`eICIJyA)&xg*dE;`L&+<L+rT>&@7#P+CFDlcJAPj`#gvyn7#ju>Zr8H>L*S20ar8BAvX-h0y>Kw-W0~Q5hjbe+QSJm743&bQ{$o^MC4pO`vECOR#bK&f3#UY$1MlauFc2e(lfR=hrt+-|b$xy7uhD@D4O%mV!5-u<@ry;)aW|ky~oFw6i%ggjQ<pkYhSC7zP<Cvt+2YjVQLN3#R1Bz~n>?p^}S;Jgr#Vz$_*Nk}*Udqc%yD&wT?g>(RFiN_i$Qla1%sybnM2t~^-3cGr;z>z#jp<M!Q<H2@TY0}7u<*d*A_Not15__R4<e=TU3zA)wg1v2+^*s!}7mb<qu@L_W+zkSRnO_(MTdrt;+^~ymDD$(Cb5K_&vR>XM=n)Pwo3O{i{``+g*{mg}$O4d>-v9{1pUC<!v%oTR-f@Wc-E;@PjKn?ATrsIR*r=XG{JKd2yfl1$CcyjTacmJm1{kMMet+hw*tUX!=@lwTo6MIU)TPtNcoa90FHgKr#c}h+aq6ITpKKUe?d&lo>T4PTr4EVz|xe~*EUB&CyPPX8Sv19I1zFJ(EDHoj+Ip>6iL(l^Zt>xxXDHfC@{H`Wb(_sg0adK7sq_*O#GzeV7J9B1j<@v_>ublOJ7of<y@@DtFhw$&l@@Xf@{+p^PQa&ET#2HW?4RThAT<Q=N>}PT~G_B7pIoMQvrwZV_%Fe|KbD&l$3aJ*lAJy6mOHJO%q*BS0@=i$6_-a&tWpy$D_B1l0Rf`sSQ^{;r9ZEpBP9-yHZ{hN8n%9aaq}^j+-`;(DU)x{!_dQ2n1lId1)k<f<`FFP+1#VU=49;U=a2^kZ^GK>cIFF_JhclT%#!|f?o$U|O9MDs>Q}+Eurx@B>iUtP(p9|G{puEHmml}K%PRP+v2WUdVFX8lYgtG%6;cVsL6$PV!ZicWdV4kJ<qW&9yD@G+oy4hId^7UzZWF0FI#Qv2>(}3q=s&9KX`tp(1|6B8+U|LTffyJL822KNy`*2$cEEO{vgn2Nj@C)4zTY+bPhiv6dknw0fvkKpF(cercP~q={Q|z?szVcHw62!2Az%BAP$Wv{QZKXYO{f{e~@BamkG9W~3eD<e}ho`-jFIWriN2fO$ME>PHJG1uq1MmK4&UR;Qb=7<OHf!<AgF6rJHk&|;8>`)0D{GJLv0BU;XpBYp63ANI6lZ>&S@>ivVKy?p$P7}l*8ZJ)4;<ZdVCR8d3J73atAFjj{rUE_N0*oc<*&g)u@UMwjql)x6%P#!P1SaG+Vk~h&N&2lyz@V%YP-pWIFrhZ45u^0sc|Qh$&F@n>G7%BzGADrr(9_trDGdZ868fghcg-Ag1K}yH#*^0sW)3WrvY6Smw<L>Ml#uv)M(~kfAbr>b~RO_FLOkIWo|4v#g<cRSS<ki$RCCw$o^s@#txDKGeT*sm4Y@3t;%m<>~tW5b0?nCCxKIprKS@h4Jr#v$z87r_v6XQ>22v$D#3nD+rC7?F*4em>5$6JWOUXi?|bxmfWL~zT)bK_8>*ykG4FkptWvqjR02kv&P}FMi8TC`$xUWQ6Haz?`c$tD3b+S>??w&>{wcd<wF-d3LD&TdW}{Gbaa0nzB#H(qskVZ=0a(p{<iWMIXYY6qSDf@baPh2n=IhOK7aiDK>v!KlBW-%#J9`GUyfc;}UmM5pKY4Ja`_b)<XOA84vyZ*EKkdGA$NTi#^>0=|V1-uQYme6+t!#Y%VEyU4-iOa%v)#Y4@#ULLm7o$bplz*sEp#98*X-Apc2g>sP3KbC(A|{64VTU4GCQ`mn?iO@f?Lqw9M2rD*+x1<1K$gaVlgrul9$==t_<)?JZQbPbHCFnxwT@mQcwEzDvK_^WwKB#%~xt}vZw7Q)Qv9aH^)>>zM9B?Q*T!0fGN`T&I;~Lxup%iJbt&*Rx{{jr&yMqy=6Xo!@L}nbL{PpG>UZ5D6F|L1L(^rG3whU!`<S{;;<ndC52eXczjr+8dN?6M#1iUii#HZC3Yz)ssj3Y6sr+6xIa&L1rDm-zEmoBthMcs+RdjK3atefHoE9VbA_E6V3!`J2Auzb2iP|O#U#5)mNV3B+;$nVbj{)-$j>czj@?5I8&|tfMn{@Lqf$E7X^@W#5V>1-(V^(fB}E6b_V^v|>gA2+AEFQBhkxx}e;DZLcoFt%r?_A-O`=cilC_+D9>K2O)G?=XxKnFKA|rPjnHs3M#{rTWgrNvzH-=^)o^J;-1rUe&+Z~2DRM>PE>Wi*<3~6;{3Zipu=?0xeXha>u&tlmjorcwi3P;{FXD2nz)P2j>fGFz5x~^(W1)MAd#O-eZ<mQW0fguRlyC;h8tIxHf;a|WDl?bTG>7L#?zQ2y#I(SvQ2}d!v&!)&E*rIX~j_omS_^|I8#fA`%uz&0-R;z#%s7>N54Az+4^$c&M?{oHtbt9u#Ub0ebomN`NsZJ{`75t4L@ZIOCeJlpM1ISD;TWqxUXxTgS=kBcw-nmcL9(~31`wnfs?q6<0Ih?pSRgrhy>k5G}EkLHt$L@>Ea4uPF;MLt&3~MJ;pDSY*$K%$^9CDKj-LStgZI?Oxx08DNY3C)<07cvvQ5PmhH*WAq**$-qUFGw>|I7O76$z^K)f>VwDR|?9uJji;G3$BbV`_lc<lm(;ZTH?Ekl0_i<ULzn|M33$&9}T;58>a<<!kalsehE4=lcJuAhGIa*({l}ru?ksYdT*K+Aa)~7bJWu7>|Jnns}G|l_UGNKzIz`3rC*N-6&yiLk>YS@`^BE!#kB3O$`v|R0&Wh;|Z`@`=Y7CATgx(7<3l3)uAw<BX9@PyYiLy#!Y_HQV}%>?gst^0WW5wf4R~!&(|+DM8iyjN}3P;;yt_FUH(#B3hgzNUj#03St<=nO|t&gUg%?6w;-9z2iMMW5o*x++pRMW+okM!hGTJ&iNFr#q}>chkzQ*c<(ryqbpQL4YwtSUd!KvfSGyN3p=}?i+1sCbXYMF$W%Gkiyz}qdnq$kiKiN3_)FEeX!RUeRF)7#k{H^t?XKgKYO-V~pn&5SbNm~=&%N_U>8e+A>g{{SOvREz`fPMPc7-IY!88_jf5MF(aEh~`gYw3K+(gynm1tQQ+>x@)K-<naGLG&5VJYl}XF{0kaI&`+77qR#jNd0SV3ed9!Dj)n_h<A9*CMD=ctCy@*<eAC-8c$6xMfA;xx#F%Bypo6-I{X4mqtXahrl3A3@=l^kE-bJ8?T`rSNlZ~0W&X13fRydDwS%ft?lg;j2oY2w;jn`}1<okWt5T-q09X7$^%KjWv+?`}h^XBU9?0lP2u_M1h1f>AF{U!mWK^^qtX$i8aA)o5b96%Bje8xbxsO<}T9S9PZ1}=HXq@5#%MyB%jzQs9fKPHB#kK+j3YbBWY)&-X5>xKRgP{4_%}+7R3_u9L(*>G(>qBq(3Cw)$(IqNbUSB;AB~hn;g*2j@2>YO0W>sh#^*>1v2R{51b+FGrz_SI}I@a%9fB~^Z1i`U;{jtLs$2v$0m+!#R05kCJe~eBCI8qy+ULogo$JZ+8w{Mvp%$P{A(hOU{z!Z#iG<J`5&uW?yF$jKZ6gRMbVKj6H6lXDx6RPA95u%hnT<g*sKaJ5REZD0k&U&Dzcm5;q%Xhqs*HPr=3uhZ;Uvile^<utWZ@K<B|9+%S9T*)=dlS5a2lo93@QPBsS(ah&oPTlnmc(#%01HKbbl@Pv%<b~Jt%K{+sWOYmyxVk>QAAPKMM$k;m^vb;dRl7GIgwfx4j+8&=pN09gDaRcKw|w~+aRda3f&k%Rh3<d8$mpJX><eM?xB!T7d}!kqgPb7fIMYJpAolk<WhU5t<pj_U0-qK-TqULjb%}TTcuPhti}KQt0o#9VM-j%+En9%Zbn(Zzl^s>zW&lX_vz-ji{7*MH@^R9{fkwFD4)=c7n7-JXP9@anGnlMM#Z#JF+Cj=-Js`f+nfwus$8PK;`nbWnKj)VmDqD3H+TH!*IG<}czB<4youVWW>077z$Rp4mRIKtb5CGrsT5eWu~~aw=&qu~LBF0T3Do7PPf?LQ25NB?sHkOXL{C|)tL&G4C2(kH$mz$9n$C@+bE&bOUJ04e7xGFFR}%SSskQh**AuJtxlXeqYQxhUB94AzP#BTvm#FF+X!XkIyn(iOUTuG|5j5+DUmx@>yZj1am}xiqPN5oLgkK^A(SONdfiqzeL^h#kHgX4p<7&4Ll@KNT_Mv{s&}FgQ>a*&XY9KAqwM9<FU)<0P0L<ZX0bxOpl6b&v5zndU-Gpxc!tG-;V3)>#oNDB>Y@h~C+<tQgiR7*$uM=osl5Oq<W^)`7J`~Jv!s@Z6j#s*_sfAWsFOFZ@dhx+pi$<7NCS<jGS!{s}p|wkzK2}?AYGUsNjEu<J(9IAs*W=@K&_Y_7RmP>>&9W)v3b0V+f?q%8|G5>d!#?OmMPJ$w=-$Gx9Trv8r1d2qmmKYg1gH2X#2-ru3TktL{o0BmBRCi0%LG8gCg!5AK!JkB)<F!2$vO#Vs5z8ifly&z<%c@$+2Qe_IQnAE20Yn|Ws)s`5GHXk4pTvF)^M|Dv2Tx~-Gw!=Xs2+qz{<v;-sBL%WoSActOZNkrsWhS93MF>4p#vTnq<CR(;><+wVp6GowWBm#1?BP=S<2`i-m-dCmgeM2}-T*WZ0R)z_lJ3BIx}wtiaxaot@eW${xnc$!Z+yEnu4cVsjt+-n_TmkKxf^E?XETZH0h|G5YeA|7?C3H7)23HwQ;k#?6$|*<ljzT0V<nQVa;x6!sk1l;ly#9_D4MB@A`M>`uCGc;_na>*v;1zhk%gx_57TPp+tM1JsZ+(=sYomW9HH$Gf`Ny>(&z=3DF${bx@<WoBJw(Sr(`x0W})c*7~zI~c#ZH1C#<S?r+x2Y;5gkzU-#ILY8PwrI(bd{h#QnnDBg7pCCLF>}*GkJIrGLeOAGo2h}5SnX8dWaZS<04fRZe~x;m`_J$)l2u7ht(h2R+yrH%u?BkXvu&QoL5e*;G_9S5_&as4+en<bV<sEy-Y9#t*XOt)XjVhJh-w@_cwN)9&~kIVn`nX3ZeHo$TCrmttWW_?u#5|8v;fz#K#TxA<YOVyN<4@z{6;nV6^lDBn=*@K_}VZ`n>}HbKN7XB`FShu7*%>8NZcZN^s&I!g$<l3jrHk+rtYHg!R*Twpaq>`)hBaXkvH^dhDz3w7MeOXwol9$ALxZS9lVpIY@DbTipU)BXXUuWmHu{XF@G7jTF9>`2_!^*j6bPlTEu1Ir!aJV<_tTP%-CMW6Mw(1^`7->i3$WmCa>1$9|Gf1!<_h05O{&E0}wC~5oYFNM$ER4Z2=dAl`=0`bi#%>QUQ~<T6%1%ntX4qI@bmm4v+ytAV^66#!`0fMibo&30VDH;}4a$BCg{LhfP!1L849UR_H})x^!7-y9>eF`juH7fcyXvQ+$aMHo1W};+|?a#DGoS?3(QHWvl&NOsWSlm&Wt{SMh&!+~!hmrAS{ZiYn0K%)D3-&3gnb5FrQ}JOBY(ps=+ekg3ojRNjg&>8o&T;o5Bxr3yd$!blt$^Z@-yEpg%+$jll^Q(+wM0|1)NQ)<{q%_&X;Jr$kl9l2CGH)6YfmBlLQvA*@2zN!qbkA1Rn|F)vXpgZ}OysH;CPouNAJQ_IOrEl1+Iq&nEYmf2jjd$ze=0~5&NA{M-d!<6AlC%YP+iGeJ*$b*pDX`d7jsKaPDYh!5UG>^*WsZ|}%s5rei}!mEymFA!WTu75PYp~S*|q1u&ck~T77pz^a%9heSN9yAc77gPa9f!3B;Jxnv3VGE6;?B^a7NMbkv0fJ3;)JyXxc+j;}tlmi1$+iV&#3upCnYoIAVAi7DjbkhN6MZ5$D-s9A+nnBis6)-dTJ4dvvP2dU<{I9vsTNx8c>>JaZDdchdF4%|CtMt$cwC^t+2}ExXqrdspA}D=dFP7bKY;8l%DT@d9%1qp}z?5DF;KxRh0xTUw|A)H@5PBZ7ym>ck^e-@*={tH5Q9AU(()A&JBwqb0|okfy=-e1Yctg|!1UYJ$;F4c_--;}b>(9ApTx@%#qfjZ;0{*rhkT9_M}WX7}S;xbELRTw6K2_Go$I*;$S;&gPqcSbKclJ9|kz+OaC^j;gcv{G*KrZ@@bbUIXUcf4q6_3O@<Lcm2K)xkypKT{$G(P)5jV0w#MOuE62j`}iI2%J<I3)2Da~H{-0WtT>K}p%|FKIkvEU-f<A$&wu4)@tNO@<$8%s7OzE9*y0Bp4^MBr{U$0%K6uc*ey)4_4;v3wy(`yg<Uo-wI8y(?TuRUl_1a^mPQih)XWx~9O~YHh0Wk6&JXYLpm@grp5{r$&J4YDYx*NbA-vN=;#?LVw3mfC+^39Fs=eTNz6(ukm4Eo>SzsXZKy#F1pDQkvX`|KsQX6VPm!XVvRChCBRV}vzGn2ckvQ<HmYi*cJp2(xLmDht9BK~WM)+t5o&F1uQb_Kq;u?7^H+-0F~~_3--=bq?>7(n*VwPJvWZt&pLpx`h5%c%{H(=flpCPJ=vkI0GSb5|CA%oy})lb1XsEk2DCK@V!6K){{+H1N8gfDu7(4H5o~zg6H5~R-lBgNMR9dNH-!8PV}GAqF+#a9!ADx?j4K~WAW8%bsET}k`fB>pkoQ2J~tg%+#w=CfEta+_X6R$3hDe;-Fu&j>_#CPWmW&gH&V;4XO?F?UM;4XtpW=Xx8Ij)`7w$`Kk=}u)U2+)%VS~d2X(_<1wzOXiHEJry)#m(2NOE3vQAL-)LDqx^@B_ds>K$WeCkI>Qe5BoSc*y<q>!jdfqU(dBSyuWI#8o>0mSw~WZ+h^0Z2cWGk3i|FN^46_ivN$cIfSP9-E?3KTd*;HMCd?of9&t1b5)}p8tu+aj;?<D~`yDn0Dvh1&dGj)If%6W_i_PzY>)V<?<=rXHf}PCE}C{G}7B?ktFj7uUI8+>NCT=*D->|bcmioah|>Z(0lq`_raC5M_=)Mdh;!I;hw$HED)bQ@t$1CIVV+B-z30$0A*P`4G+9}_~2^?cK;W-s3n*?O0zAZm$g#~@{B6Wf^>p=p25RnrA&|QI(x{RO5i<AXWBVs-TcCpn|b1CXeG_YN*p`ifUHhwpXqAg?A_gR_{o@nt#B}77=iP1&ObL>>^H9nZHLM3j2lvJnMFhn?~8|hQTPCbc~PDNVakF{4AliIOt#=KfCs6VI34Udy!YtdT|4*jBoSYAFZ|xS^l<Iboy~KXHm}{-c<=zv9DlyDaqTIYZnl@*na<o+j`>Rq&dx)7U&tznHv-*qJ`g3Eb3IWJs^CBIEjlmo_VIvTDsbc%zmK^~cC{3=QO((Q1P(D*TPq7|f(0=IQwt8?Bo_7xJd~b%8BWQ4D{n7W)@$d4^rUVG)ugQXxZk_Vb5H&7<Yqgc!t9RTne!lTcUNz-GYmVxnDf9Wif#B|laW!G^eOXgQ6Uxb0PGBs#|Sp$@GE#bODf$)9ivU#M_=?6P;n8z#+q@BdCvu~I^c#}b2aH3FTtAvxbFAGwP+}YlMge~c$ULFkb}W%RA|gj4e(eONB10ZPRjjs3JP`}KFYno2Mn+AbduZKyVoBwextsp3sO?Do1A7mSYFDLCp6o|IgwPYx5%0c8sGenYWv!EEJ$*-aA0=^QQ1AwM!m1nZGiVGbVR9UApi!zCyyN3`Ktrd*fx*4$y20I&nv)}N}DZx>P0`JtO0jF4|l_o8dIo=qhSp~RQZG~y0O)uwL}2zG7VliJ@dxK{YQ8r53CPqO3r?>e($=~@5m(8K_+jPl1ud()a$qwk<w1q@gTe;yNwCZ#BDR;sxsDu0WMZVMVpG6)<_R_E-LpT3#QCs`L2Jny1x2}%!-Q|lWUKCVMWX1)z=qvZWL-mGPjn=rFt}U2G=h?gq3C&z+{9Ip?6`S((Ej*=a7^MvkRm`^+WpZAZuibW6n@ms!L2@9%Nd5txqQDXkTTQch^?lSzG-M7O4B_Bk%d?{+FeH^IEi-E!@RH+nY%x@`Jg=0RbtZo4zi=h8gv?N<0?*#8}qBSZ=L^Y;OgDgI9l}ypNd7)Ubw~HjjmPtIb-`dfL-u8EfA_`lV^4u=DWYy|3>n$m|=_j_t-b;nUQB&no2j@+TMVtRrN&!Q^!$Zb_!B?=F7FSkqdVIPM3iH&SQnSsiM|2Xmp!TF=(`Qz6};YKaR3O0Za}qpx5al@&#uTm<_l`0ZVOvn^~4`~ad)(snutZ8|xeo}N_o0Mot37Ca1u7}xtpbrf?5!*}je1)T^KpIKt4mse6sL{wKbY_Js+jEij6_jcJLGZ{&8E+PZr{)^1@^dfvc`uOw$1I&Un^8Ov~`TNM)?tKoYIJOrsyv*Hiy!$t|E`a<vKGzST<C%_$Zj+{yC!T-ls3>us{2O_vls-nk&(Irh5ohITZrTZz&aL=0%kPFk8-<F5*A;BnuP;il4F=BW%<$-s%y-y(_C81`VPs>L>G|>}5iSI_A$%T+hmN3KS1Y&c5miacsMhw?juEYZfT(Q*J%Scyg8+>YVcs}s{1{SOTTqSP!m;`2k?F9(Mw6&n{VHjJV+2LD8dBd3O^0?M?)>YGMQKFCSvZtvZJ;2cI#~&zZf7Az^`-eC1nyg=Bx6bEuzS4NEN`)8lNy=nc7=XHQ_h+<>W#=EW*;`)VMKol`RKu8k*ruQ*#yS(Qr?MSj|STvc6gzxrk@UAdvS5jm|gG-+sX12m|5JUsRZKTyxFI|&~3N5fS!`%8u$uGd74zvZgy%V{?b?dh&-jf5i?AWM;Z=PTJ1Todpbxw*%<RD*>6+T;eXKdG!IerC3uGkWmXA)4hhXD(a(LJeSVb1y<3Q~sKi;+H4jYbXgF_6n}->6F-jvTgCSUg5O52T2wy<qQJJx*uUG_n0APFc0U5l3TiwEj*+WVffS5KkTw->UVlva=QMv}5$dG?*emH`5Gp6)w&fAnJaK^z_Ivj%A_eJJwJ#EOC29C92Fhqq85~eOrC%Ms4QA{&^oEbPo={h1FNN6CCJS$1q=%Zc{pDoqY&<gqqB~msJ6U`)jd-lft+w1q=U%&9j^{Z#Oi<D)uZ6nN()SaQ@_yP}*DWg{f2h1-gahOGi7%;rl^F6|#F<4G3#lc9`Jj4%1VC&xcn#ICgdvp)4o1K4mZS`yK@ptGcwS0Qx#=9HeKk&|e>plIm_xnre$HZbV+2-l2J-xN|Xhn}NDm~5Q*F%su5WlHyjRGjp%m}6-7esioQP89doA2M>{Oki515S<P-R}3_uRZ>A7$!(GSYP>4wXMTR2WGvta+y2iZs)85PC6Fsh&Y9`+%^#QVy6m+5qYx#k0cS-TLUE7nt_bl=AF>J#O?J?$R0900GzN}6|An}8ejx6(XYLl00#K2JzD;Mm*2pkf#+{{7q82sr1xZH{q%X%b6>h@hwBzZ^xb=Q?LD&h;DH`ex>$lx5tv+m(Yr28qNbA)gL&ln2%#Qcp`SA$LvSv?3w+Nd)3ecsAOZQc)yv#LoJcqe(1>m@U#@Q6_}nCN74^bwzZRAdGaI|p;p2OJOH$N9-<#i;qW<2u2?7F4k4*uq!<e+6rkyiyRlYRa2(BR;Nj4JQN|1h~ofa@HtR?sN60NvxC8|YWF=b<mV?e#Jq;(NUT*xt?KGU*_vVY&D39L*|tP0*9`ELULsi4<Es20z>^H2VSYA7&;8C$#zb?G5Q!8VAV&fbMfER$3B{B`a{PHI2kt~@WKHS@7_ptD?F3}%N^eOAJq9s5rc8Enh*M*2xh@!KGqWAZBI2`3Uh(Iu#m%bPSqKT%mYRQ1&&{gu8tX=8rIRc-l^VLG<!P>qMg=Y1>{v^SYYjR=ywC2lG&;p}0)pJa2%%baKwY9N%OnP>GaY1p9ynUMXY`0JEFsr*R`Bu>#CI|jD=Nfn5_b%e$p_^6xb3~PC=Yr2At_QI1vZ>m$FL6;40+`O&yB_|!`jli9xd7#Pu5<DwpB+T$_BO`hOX4}@xS*~BQ!6bj(fvK<$xU?-vI~1q=7Zd54xv^NiO_pT2+iZZhVLKZm&>(JuxbxIIeTAvHyVo9jS3a;c!eAM0RRqfjSz&Vq<hgIFt>_C`!(PV6ehrYec0B%ewDnQd320FaDL2VfNYIMjrnrEZP1*PaeF(Fa1@z*oaW2`f<yp}(M<cbgH}?t_xT{{Rx+Q+a%i+x!m1`+ZcDoNc2M!($i-=~r<ngdG#;O};Yw64!>&x&R51&K803W=b{Q|p~d1jGEi+BDe{hxA10H?R;$hhM&u(o2m-DFcjK1%zC4-h#kBLJ5D;Cp0QlDacOD{v)vd0)H3!K&E=14E0T0vm5w!g|Ckn>Vh+X7-++*}T09$0!a_)SQ4+bj9#a!{&HzFxXkbiP%=SL)aw>WnXaMv9Mr!_8&U>9}52An141k;6(W>_D7HYuC(YyJRxlP49Y>mr6a%4pagpj-(@-i#e2vJJtZb~`;$8>09o8~;Ap4<qAw4HSv46wP)~LAec2cDC-o_w>Q`XvDKrVaLif=b-|@}&Nql;C?a`%m_3pY|r(?m;po_V|lf|}>7N=l6fjydJLHTLvdd&IjGOa6KjlcG1c4vEYd3EF2yWUxJVz3yDbbJeh;I#>6?3^nWnU1-RPTT`A0PRuFigf>yQ3xUjMF&$_S7s|EXYX#3s@Pp*yypzWxLTjVV4o4r;)9OR<tjIuNMC#Tj(6n`sOx=#Sv|TJ&tWL8y}K0+t_Ex=!ZNPHz|eFlTj_!n(d8!0+lAG92*-0&ZZ9l0;{Gq8n=u@@_R3RIp6QIbBGWU=`^;Nls&H#&=xrh<PN!DEU<DWvDB+}%BXNINc=h?*6(^JW`EWM%b0aX^H+Q|W7lD~-x8}U_?{D0`8-96XQZR&RF+{yrj399D4)p3@cMSK@wcIM4R3R_pEK<er8y2;`4vXv*Ytj<i`Os2FB8<WjE!FxyS^9o8DrTSxz*?7MfT*otN-mk1wKnk&tqLZ|^Q+n*u3|MyigWI1_q~T3U%bI@nu`<}-u;i)FJn?%<~+@!dTc!XV&mCa@9}rSq31jteDFTt^ABOiiafch?u{8&WHOb6GviXEFFzrx1FNn*QwDLmk>Q8WO(#>NrlaTRRDKjT-*xsEYx4lQVoi80ia3MpB1|iSLdhWfd8atjiqVK!0CE^A{1Jz9vi~A2UZhoHtnLjVd))5Cc(^*2+`*q~JV5r9F8nw#e0f4s4~zFA2F6yB52thKkz6J+Xm%zyvLl!MDU%Ne^1EBY5WFtZwiMe%WLupQDi28D^zSqA1(ksT@aEI-@k&c%Am#@t<XFBi%L4f$xA^gyiD8|GQbT=ZA;tq!3Mlxr784p!LdEzjyMmE*=9-oAc8s17`@cuC6!{|FBObI&KCG2)HK%ZRMXq!YM1*#k%MJbxxWuTkmVlQf7XdAnct15zfde*XcpY@`uLNc8!C%3KO&3+wC<3v-3o`OHHkL0L8EDrr0C8#F(IT$(Z5v{iYv#bgq?RowF*VYB=_ds4Ve{goGJ-<xq|bvG=WhalMti=}5~ROZR9bbUT5yK|!bM~+v@DsB=M1(%P@t(NbaL7Ea&<3W06~S_WZHU6JH>_<l>vfL7r{N*%p0VxaT`vI?q9|uAZtmb$zf(Qb{c>SrVkX|85Lj72eJ{`77I65m>StweKm)0A4}nVf5$uXefRp~?l<pk(U$suB#ECwk}z$fp%&^g#?<d=2j$!*XAh|%`HHlG>ii?j9i&Ho65hE=cjO0@LTf91L&$bUu^FDjYFg$&Z`tL*uvCFqtj75><g-9_lRlH1jO5apFgY!~!^w=GoR-f16v<5%DE&n5NN)Y)G!dQ#MkuzeBr(y$@I(lUJco&yvZP9Kqn@z@CxvPb=@_AABf*y%SvRtg;Cqoy8q++0P%4}gh8K#*<n5WCI28#s6<Ad3EQm-@=~N)kNRu##zfMS{z=JEu6|X<^KKq!R>);%yM?s+0l&F2NSd~49)|)r4dFMVwDd=~f@Yqe)AF`faFkOi`CrSmO7g>{#G#TgeAFQ>+L}X1-&~MZafgY;IGsm`I@)1Q{q{2kHN-#ya5~IhQscEqgIB0|x=fyE}`Grp7wh#nj;FF@#G-W&Fd#`Wi4U6o(`_y~)ZdaW0|4um~q(4$XBl|xXbiw`c4@h0XwyA;d{%BBjV<6|o^K0IRALBc!#9pHXZtw1Z#_-1{tq}Pies0~UKMRVh;h6|*aj$%e5I_lbj~f<M-bxWmjY3%^z9kG=v<LQ=$g{|lsDTW!3TTA>bFr_`{%X{HiN=H@VjHN=ZMtT~qM8W0Nf|-dQzF#*NIy~Bx7OHiJFO-d765-;z*tT!5fWluKF5=~Y!(mntDJUx4a4T@2OIZ4XUhGxM=Q+hEU*$Uu=xXq%AYQ#di35$22Y>m+nAflZu*smo%JtQy|?e8N3I|S-jxSyt9Ld&`&0MRbLfYD{?pAHpQHEP{fig~mXse_4+JRwiPbOHSMPcEKJae65TQ`7hFRST7MTHui;*>ELf5$lcP0$s0!i&$9}k?H4Gi%Q#5rQ>3JmDp!u5^iGwWZhdQX<uFF)`;I~^YHJ-Ew**Db0R!VwW0ev8>Yx}V(Hc<|YlHCDzs(Q+tUL~66BdX*|`rMQ3U3lwnF#jPmFS)Z}y7kRAt9-pFxRhN|Q`f1&i+T#|DA)Je(B{V!zCrpQBDX~|fqkG}lLSx=nPmIcIp)_^a;t@N14<D}*fmiA$X59917t>Q6uRDS+qN+mC1UBbd)|@9fXYxeCS(*;bSYfZAP6x)(gwvN8m^?q3nqj-!lB)8E5tS3V7AN+Ld^*A}iSowA!V|~F9m@4eJNec_@55)^2iMk~z2iMxanken#yj)%=DCYTe%V-ff?3|7VQ>BVOYg}Qmecl!C+CkXXpsqo3Wp^f$K!Rf^_ym!GSlJ&SY}%I$GA@+a&w($M=q^T1!9E-%37m-WxKLA){NpC=f3eCz6(b?U>KO&>y78$nM<lz&bQk_@kA`mxH*lx<W{ON<J<9gTvc~)%V>nbO4RVf&1yLicDt=ov(jLA_qyU!&cgPi^uqj<$f(Uu{Va=JUly;?TqYfzCU$g3E}PkM8iiik6W;=+hrsVAE<AJ57y1wm5hcBmx+@s?W?ZL8(Yk%FH3ew1Ro4{;u)WZ42k>1I4aGF!{HEh}+(ZD>39_oqRx5M!ZK6J{WN;w;+tQexAr-WZiBzj(7|XCA@QP?xuXdD(+X+?ZGw&q!u*C`6x!)Ona*rM!1?cr+(D6ZpOeqBHd1QzZui2EyLcmCyB{X-qB`-K3C+|evyfj)F>7B9w((LGTlsON_&KH;yalLDTJr08Xp@uAK1!gIm)#}F!vZ+Cn%xuJdW=ieOS3!Ca+Jm&x{gPm;TwM~SU<IKaRo&vE8~)lJmOMy~f&u2#vB^N!fzjz#Ih4B@W}<fu^RS*?0E7CuwiLZYK=D3&C3?5T&}QP!7KflS41!G{<e6e14Man;-rCBAFSiW-q+I<AMVyy3<5>_N)F=*WVkF-b6)LGo`Ame1CR9OjRirIaHc8C4D8-Xu=0UMpLKP!Ho{h{pFzM}Tck%vF!H(6v+KE*S_$KBUKMS5>cb3b*Q~2k|KJEJS%?2H*#ql?m%nhk;Ae%kaZ)A5v6YE}FELN%%BHNz|t-eg(z_xK<K8Lv>4yI9aPqZD`5uYNU1m*oEpLX~y-v{POuJ!VqqmYrnLDo%HTK<Tw5mY&fxB;Y5&AMXpIB<T~fqe0TlizLF47ItFb)-J)!+<X?X2mU`akY5Nh4-_88Eej-wpEC0n83v)Yg$8XcD-(RGP9!a0SsKMWH|-=Dm=#kWtj?8mPSp*1u$5N7^`1N=d;>rnd>mvYV@fYQgZwe=y@Jva{LI}S;31Op%}*ottnTuAgR5H%Wh%Y{uk#JCTbj&x}knZObLj#IQ9shk=7gn8X4u$>INVWI^t9oYh>6NvXN%z$8Amqj%Zk%=EZ4V93U4W?p2>$LJ^=@ws?JUkPcSbE%hLtjBq$$GJy#ai|Ck$kPhQfM?l3F!50|gTUc<O#;xD=4c9|SeyoxSpGm6*B)%kSMY&9I)nFlrom$f*f$`o{#3GdkZg7W%U{PDEo@&7kl(*K)^abNGx>6Q$Z=JR{*7p^x$_x9Ied$vn9|F~UiC>PzFA9cv;wwV|M)=>sF6E+NHUCM!*dD2I(A_Y^rZThmz}wzRg{Vx7nkoZN<GwnBdn>R@R7$3Kyfse5T20WX=@x8>6t|Faub1aVkL-AXtazw~c%hVdPQN+vEV_?yZ;<^2*A{U-B}G$($znKR!#|`_g`McOq=U%b8ax-PmES4?sPq6i+ANlisp};Kec-I*Uuq}_hWU8-nub@C?t}u&zC$7DSK4l)r&kH%fITyZ_3+Xlzmrf!0z1H}?hpzZ2&agUw#>mp+5`O;I9E)^5OnvjAGM{Ug#as>pWZZYw!A<JF54Y$W+Y_!kd*bwINguGv&AnrCLK7;pN!6_<&!zePc@Q2Mew)pY&>|+yS382p+>Y%Io(?qoDm!dre*a$x$Hf9bMy4u-o+KXmHy#hz0beIYgUx;q~EVnva58W-|Cd{D&4!b;VMIBw-68oCHX-^sTT#)K_wz9rpE$;9GxMH{7208r)(=bs3nACidlY%MkL)-D=X5opn{><y#|fkUaFBZB*F*FEEE`dJS4N2b`vX*8H=4_dD^u>p1M-YoglI%*!Q7d)qC)G<IlIvOJ>|*N08Il7Gy^Gk^Y>B#d~$|mtZz5S{us%fevuD@2D61&GA{Cmy8fLA*qfm+4Ph=MbUmuI#LEbSPJZxgm6s?%MPdLh{LIEz|<Tv&q@B>X*JE^Ui;N#yj^Ma!wrrt-%}WlTL)*oSLly$QpqRouPoa-WcARNc2GH$v50Zr`47F-6>fE*2AIt3@ps*e7g&TFb_oP8L8>8|eEDNg81EjPS^wjTcl9kjMxjjH;X76h>IZKEx_<f&NzSNE!jEI1Bd7TdvYSi(7mU<ok{LZv)1VfNLH43tGZ>7FxG?C1cyGzR;?{nIL!Ec+0W1>NB<<aecZZ+7<9+caOg&XdCFoZgf29klWHM>%N{_1JE>cd4!pX`hU#hY+(C!|~(*tuiV@fGoQ}uiPb>y&``YmN)0~&O!n_|%EcWYQ~+J^YFx1>v%dTrdQTVngfyY!we6>;isC@Ho~nbdp2wr)tZ7)_3hi{6P|v@)6=MGM5tS5&=LKCPMODs)8Zmvix?n>@woR@CjijVOJsZTI_C%qUO8!1(>_UcM4g*jU{_T(U5rC~f++-YiC#ZWjBq+mIznp4dWByocjae=}646|f~#s|95;hPx9gC~fXe+WbRu_%bL8wlLj4s<~P=-L0tQ43&HzU=1;pF!y8bfk;WT{OCUhq6wI|Hk*h+_bfZoVj-E<f}pc|nnG4U)oi%iLSUi^!+yanU50~LA*!9?!Nc7M5~>~(2PB<tf))ImP^EgO)>eX_TXy)xbKSo_<I4{q2btJvgh_IkaA*W(kWj~*FvplM#}v0%G+x}>mHb>)-t)VQT67qt>a9x{c$x^>+Ugm8W>XWlF)vSyzWl{%_q{)ejA|FQ;?epd+4g*WAZ!yppogcul`mc(uT9^<IL!#0W@-n0XGa*~S}|)Bz<}k_Y?iZUo08GsyB|F8u72TtbhW#B+ZGjFI_?UISf!CK!!Fn2Vf0MT7(SLv5)Swmy6^mP{loh}mb_aJSt=s$^S6LvtzY{y=3Zc7eC$z&toxq4&@+@V2TrBkuWwV=sNC^BT+vu9#T)cLeG#=hRAuA*SL-X!Z9~_c4f*8-akbSq4fV(**V|MPB@P8<+&fCV^}xr7{%<;!CI&;Qxy`vH!_je@SZwwd$V{!(Ss-mU22dgG-m!(KB_2sSERqfjT^fOIt`@}SWpiXd0?kHY46E<LvN94CHU=}8oZ>&}OC6Ir1O2BXPUC9B&<@S*WKWZ5-HXF{b2CP@)ZDIKvNR2cAf^gq{6OUZVSprG|7JVL<a=$~AOj3(t8GG+M~1}EG*(u7?sKCI3RZUYGT!hwx8g*owZqSg+OFjh)m`fcXW-~~j1DZ1mhluQ>=gJ+y73oFaH=Mt4dLf~v&0uus08>|s0)NZCbtozFayZcMdED{3&QZ}CM;zX$L7u_Sfn<<7|$l=Hc+g}<EQlCq=Vj>uQ5Cmg?3`Hbobgb@77(uH2lTN@7EsPQ}2@n`N9i&QE*wFa+<6?`h5M($K9{L>^|f!@gah|>2op6JTA6v^ZhM`nYRTLX|iJksC)@h0<7;)DeJ2@0(Ktn>=_w&++pF!BX3j~UuN;l{k?&#oV@vAJ;DbP6zBTHWP4ck{t%g1oD!~_uFd!{%5!0{Ila$5_TK)~L}`7CsGL)w>}(k+l7-;bc$s0F4#lt!W9pdi&VVT2y|wIp{w34T_A`P!?$MZ5mJnmI#p8>I<i-sd7Xt<03W%>jjl0(#W40QK`)+L!kS%_Zj~Y!b+}h?}a}+DdzS^%ofL~nf>%Le&ZaQK#s!`Z_d8?1y0MwVbSlcuD$YOsIY`|G&F~U=ZHBfa<tc6-)ZKC_YEQUM4g!|(x>drXGNhpwm(9-GmwEe#PY!H)nv^g0Sft!0oCPM`s)7ep>ACWBtUN=f@)=nviZ0IZUSUWw=ZdA)aux8dl?-<UOurOBQ8mWU1ww(?QIwLfW%vrAAy}Nnt()!c;TQQQpSgeMiEmyaZE0t8UTReoj_0neMLKOGmcR`{~IFr+1^~{?oeH>9D>IE;D^m0fea!=l96l+BnLVO9Nd-rztBRKfW9P9<+(0dApcOKp26GrYml1?SZ*zGCBYedu_PmgSJB$F@<=4nwQ6ILTLRbw7@je50GTH5}<@S51p<He>c6<UYr4V#h8V~w>;)?8?$9je$1;R1Gmy*Ikm`=Co|g(7}Fw4cxs>P~3Fk3%#&BK<n1Y9w%Al+<f?Da=u)DSh$ka7e4Sl5TCW(xe28`wqVP+Tqs<J6}6`aPNV=M_EGisexGvBR#;5MoPpaiW!7o;156!Tb0tTdTq8c7sGN2i;Irtc~N8UfmaSD#4C0)GH(fw`|}tGo~`1X8*foQH&&nJF=)2(lcGe#-NmktwBcm8@NcXR#UKjSd8JZyMGas898A8QB?%QV7C+@R11?KBQwIg%8)Qe=K^`s&1LO04L6+D@WiiS@R}k~zIagr~p5N}i@nGZf_uiH34i>0tT%Je~i}@bnN?&#iV#hHS3F+(G-WxZ0mdVZ2PdC1JgZYN2u^SvLrR0R(`G-m$SdOjlJoV0f3WeNack`|8=`-HhZ=KCI|FH4=2E2grc*{4uFWy{x`kvZ7R%Ks6ed5hGWfH_6p0I1aAXT$QbfRKe5m5g}g_v^i{8jJmPgps_CM4#KW<^B7CV{NeqI}9S77D$OVeAWLjd56|X#%c)X66SDqRb?8`C`PhML6TVnlXw=9u<iOFivtVGEnm55dAqceJar-_%hZFaP6>mN^s@LAw|X>nx>3));4-<IM!4zgRdDumh~mjP1+I5@Ve&jX)c8#Ci8VreNMKx7XGKm<H5Qb++3PJxyaLeGY%F8Tq3+wS>dua*eE)xXTSH(t!zBFV|9x$6+J$rDEdLjC6=57nkJ7es(5RN5jHp}uu^e%a;Q><IprcU%@YPkA#ea0B`${Ip(QH=v@#58Q6CFz5$eFUaYAyes|d~4w+A*K(sI_A*xaDJ7%Dc)T!vdj<wH!aq9i|2RHe~_sL1pmtg3hltIo0+Q(g9FSjA$AJW8-4`_r~3lnc{Zz#vi?F(<P-PuCv5<6XThlghGsz<s!t4EP4x%7_iDt(;XQH03>8$To^ul2L=GqBwQ+I$OZkeAibc>P@ef@D;~Tb{i9k*z=>Al=MttK2)<6iv8pxcq>_IEe>(?ky?$~Vv?O$_zN!N<-_H>n0m>-R<4lIS<eg&Vh{``t;Pe;Qk!Iy7jsS5EyFRjS)V~YY{IybRI0nPv-ynnqEv_Zw8$jIUi<B|JX2yLfw)95JK}mVzn8#|g#+<E8UZS@J1HmGjvJD^%d93k7uOm5W$+sZGI3gaI?!2|0gSFI=2`56uIsXX{PLmccw`UEyI*h3D0;A^P3%!$hp^;ODE7NGq{Tu_hR~}SlEb#|{FqKSup{C*)asBbll8@=S$O3zoiYCs<zGPY5(vTC^ZVUvkAoa@3a<}2F9CHL)G<Q?^^v1{4ml^=r!)pCx)?NpA+Q_~Brl1$C3UN2A!HP=WoQBYQhtbof*5S?gd^5Pw;!<!v^*Pj0*G020k6NgWfmGf;c%Bgq!46};U9=w%}^YWPFeugupMprYt~H28xOu%U*$2_nA~H2M=k^`*W}LGwg?{%OEOJ&VG(K}dsl1!E50m##^nCCt=7paJjj^7`*DtH__0&`D5f>H3J7MM$Sv4oBWpLl%fXA@3f>=)86sHBf0z(39x^E;wTVk3rb&VQ%@ab40aAhzJFU*bLb18jKSf5>kPPUi!y4(F32z&TJS?*t{5XRB!bFqw7kKpyHNAxbvlSI6cW|K~ZABK5L)99Ii))Sy5yfB36k9IW->_FqMKe<pYMGtC+?BvM>g^_Mb0&+a*93nF?K$^{nsLi5L1tctv;X>gchJX%eE(IuhkniOU(IhxpTUx_I4ZO&!orAT;LIIv6h!F=oz_}yErKy86`X+OC@?!F$3wd-E1VAh0OElP<1qbNPsN<F1_J6nSXE%#HFPn*vJL)BW5Rr^Sf(My?124kJbxFBq~}(=EAN^v3@;K=J5=fm5%<D{y)bXeNPBST=b2%LOzOe&n|I{_04Ed<ws9*CpXz_>AKveNdM>O3Wro~J9_)dw5?dMl)ZYE#&GpOYet2^J<unGX^Pap1$h~><O83@EclpcS?vtUgp_oS@?*gHFw40skqft|tD^ygT^yGRj=4vej_KmvDVKYWyfr{1}w=Zt~@jC;#ZL1O3$ijAj{J@?n)2}>3Lsjx~XfL=Shn5!@6UZ(G3s)|qMXyU}Oy!8%XnYExso<^m(0KL_*_H?Q>=Cb1k7}9cNj{BO!3n#ECJ&~Nk8*+RaX$@XTk~<h>_gsSu2)xuw+N}kAWV0G0`@^sLGxjM0e<B3nuzqjnh+m24ip;9Z>`@mfs#2^?ICAo{``Nw9cKF!gJ2R#)QAcPx)Oc5z7Ug9pj6ss(U1{lGX!So&cimo-10{Fn`+CA?w60NG+(cXU@W0KD}YQK;<VMn93IfPf+jjLm5_NSSV#X3#ns>=4DoKzD+g$7z-k@LUMi5umqMZ}bb1I5Guj5}b%v>>8XSSv3MKr8kVm9s5E>7IB4O;Ki#bt=PkhNHRHu~tN#qaJ#m+8@m={EX+?x5cS+!7MHB5Dst=>;3<FH-JtaJTIG0ef8dA;`~wn+J8?a2pHDADVyP+7ls!FzUwbL5XsZ=SnkOeO%I?mKt9PrvP+{^R=7n`R5ETtiThK>3GDGr`D*G!dxgKbVr3F=-sKQM84y<NM(l=ulFJ*<#7vN~$~?`_j`?`G3KFo>uKTVLVD<Hc|Y^^qCCd4_3bLKZA59wm`f4Uh7?Tpv=|!T&1*ybq)ekm9F)$pS`1^@0zdITdo=DcmW9WLa{1iyq6tv$^pD?S;THtd*_~LRO`*6F7a99j9>JUGdNH!a)3c0bcI@^<S&3TZN{ZXRjO6s?f;;FpxM$v!I(zwMUMb7*G@3``>_0vtWb2QVg@g+9``F4;p=7kz#n&tV84FEsUgub!M!raOp?2dcQ5~L^Mg+qvkILfldGyOE<WlGt!{6I3Y<G_<MlP3lHJjK(L?S4THHt7MeZ;{)^_=6-^twWW4@?5nL|sU$K<6vkkI_i4Xucod%$fj!fsN70AZj)1cEvEc*02sfDU^c2$0KS6+>emlG_7)M=sb%Qj-thptAAw!TQP<-H*_dCv@c$?%Hr`#VVgcdmbI9V*Y4evW#8({wREZd6lSI=*O5hBH!H>O!v1+@ZC!RF=CqzywD~AnPQt|)_okQ{lUNNlc}&%HJ_y4=tmu_{S!<J&Io`}Y+?F8xYTRaCB^gRh&;ClJ?cTMfVUPaWeg{WG3rF9X!;gz>p9?NfxvOeA9?06jyX7vfAUk#|G=Zk+Ugx%;B>D)_MTjkXBF?^6(H~4`S&d+mVd@G#Xq_1Z-VSqypv18kNKeh85M?v4BE@KuJrml9w_4f=Opoyoh14=IWT%`Jvnq1tb{ncFV2-6x6!Ip>$Txld#UQGX9>sP%>TF(1x$!>E{kxlb5E7NSW~h6Jw^W)lWYxf!EgAoKW~e)pu$A92@@(bumrF3xK-`L`F!M;jV{CTKn^<XiaSjH2|!tLe9fKW0?1I<|1cZ9jqF>o#P}?;#5b#GmbX~q`3ts-k*w8av}7hV?4&ol*$Z15qDrd-;}8@W_oDCv6h&y*?lY*ZX}Z<8Sm5ST(3<!oIHW%&1_eGMHU}oVAU6o)C^6~1F@#BJ%I_6Yyb+pi{hvbruZP^`cvGLjLlleZ>(&(H;ZkU=v(Q*_iY=$6<@!G8H``MS!>l7qt+u-`OxIzY*~$sGJY0sOOAYz1qkP!mA7-s$JLiC`&XFRBPQAWSq2<TD?<=~pDnuGVie%tn4g2&0eUnQ;B*zsFu@i|U%t))K7pTad{RU=^8M2H%UG;)p(VzINTLcW#sP&+<z?Fd2y#x~~)ffCnnNo-=0$$=rEAgn&D|YA-C-Rq^;v#gz?s$_P<|U$wdI{E1;zosMu%d~M2cp};g7bGKSY9&VE^^UoeQMbtYs&ua&wk3VR7wi`0vIOBktyQaK<_x3wpdW{QoAm))Z>49F$GLeFjdpAHHvsqvgQ2NZPtey#d6s#C!=(j0F(d&4-==1h*Z80KZ{z}fAg)ohVnE%yc&u>W|E@@(NA6h({fN)SJoSdB&7~)!P!a^&in!R868D>QfO+wnO4$0(WZcerYbcaQB~Qmw)z%xvC@1pmBf^E>Oa%*t*%}sX{1ZsN^||bQKZZkazP(f6WtS0jzJM9V-bDAf+6)4(1lf0?wC}{?hAIKk~}V7M$45dQ4C!BrSl+D+H=quSQd7fnL~}M?UiJ;Zmn6zjBNvy4+2GGUA3=;UBf1rk2fT^ghtPm5ScFkdwyx~#o;cs$p(JeLYd6ry>xy!NB}JhEQ-mR0_$S%BO{Ff6S%~!K9wm2uZjEh6jl|Wu>K878Ed6wSrwYUssy(MSHk>neMVpjuxQI7vsL=ry{r6!0aO2PJiqF_{oxN!&aXYXwD#<yjc>2>$a~-arF-iF^U(}A+*ly{5}Q!3U|le$`C_ZshDCK3QQ2<=Kb%TKCTgwj2RBYy-^aH&5>*!IV<bv->qwOMfstsE6eC02V(}yPDBT*MK+3?5O05^{$CTWxW&@^5s5O!cZm|}#LqBc>zJ1>zi*#f<><D|Az~kqC0z5W@#@5VJ{ubs@`3FwJ5PY`=Vim1=jZOVX-yrTA-{i6C1$$71ps@KX>^h!M;w+JIl#!``3Da+_azNP}(}CaJT)x(`wLx832ZKBk80N*0mzN_tQFppB1m13WwP#a7R(azvQXcHY*vJXp2>kdP@1t{UD&B=Zu06fMZ0dGxS!g$WEbE`#V_nI-<!q0#4Qk~h#vZVaj>g4e6x&9p%LQD60vo2-Y!;VdfL#fv++J$9c_<@@j$|~|SG!rPwJ?5$+bjSc;w6u|uB(9OlAm^qo|3g1DKCWS4d^Y}oAJ&>wNi6iBbXX|0pvBRU8ok9(72~uqj-fM#P^6VGA&5tt^wJXA1jJgcJQRIA(3sETE%ZCn#F3h<o<fOQ<bM%i>{+wKm}7ZftGbtZ06G`Eqq?9)4=#3N!d)j%0<4-1S>qWwqm>w*{fQ&-jbDz>_%itel(u!)LOskxbAP=SURo-%U+a?+!KwM$y)W!<Ww?oXPXlnX7AC|&i(@W)2cUPFu@d1$9UY?ZW*UGy~!M$(sH@Xv>ZhXxCc&U8;&hdH?7qz0Yq$dNkIYh6qq;Q<4k9EmLj1vT=b4;xC$&%aV;VuU@+cQ<STa~u4#Bq<eb<q6VB0uvpbl;@Px3}vvKB}6G_xl0I;X~Ldv5qqBWo-$7x5w={|5%G(YO^L)k&hx>aoVi4N}@o^4iuiBy-IwipoCb~$oeGb`W>zZ}?VcghvYT+jO0DPBlzqA$G<J3=4nP%`DZ??E~2Lx2z_#}fK(cV0dG8vLuw%Wol=CTWn7wL?XK#*et=q$*rnz$?`7YeMUX`mTIBGoG+rR^`f4^M-1t<q8FE{%Et~TH8v98;wPIfe7%_fI@2#pP@a*#;f5u_rKuJxn_s|rB&3w2vAjB1v-WKO1bRTLgr?HzT?z5LC&jb&T55uw^(jzm;GmFsoP9;%sTC&zl!F}l~;KT98s*v39|>9CN23u(61ptSc9vpEvm)>VVC<l(LlpwYrfbpwX8>>=26pzqV$gj6y-6XgH@0Oswz*Wokhh5C$Hm&QC9=53eP^YVunidl_q+c?J<YcvoT1~%3Q2z#O0)cpB{06G}kbUf<=&r_xmb%w<N|F$5_FHy$Ry!0YKkaAC(Wg`(BI1gWiS_8pz-z_Lgx9?WI`UQnONPblPZ^{xz!{IioGEQzq%t(fvqAx3*9QYUL$mvCUXkCW0NDxC%3S-$bhV?HWjyEms-=6IpFug_~t<Ifd(|>P)c>@+N7-1-TAA=!MNqBfhpdg5aWTcZ+bT4N^*vN4YbjrVwP^FoMjWr3wP<E$srE+ad#UFPPMju;)2_xB2l_77DYjTjnBEo>Bd+1DC!9<BkM>UkDQffo1j@=xjsjEC@a)7+y3s1gJx^SZTS=>!`cl(`<q`Haj)&TJ2b^e!QmRP0l%~J)0pS`q=&>N#|gv4Lid5H&~`EG-XCWY8s?WIwD%hQ{)E*omUuA^iek`g#%s8ae4nJoiE5ENiB%5EK*}GwQ~X+@9liV(Z%9wWHM@Ftxw1VcO2p1n-tI9a&WMqifrz)2<kz{0yainTt=H+$Y}edyC)J(L401{O|cU}f*00v6xEZnQz#$xZLJhJ1_1=+1Q0fTUA&Z%qcJv&W`2UH2T25u+vXeSib<ZfuJ8;voSuGR!cbI?HYd%mo0LIKuBL3_*gVtGPymr;)$VECfWv)p$w`l8Ml#uv)M(~kfAbsEyC^Bzv%{(EaC&5_hT%#N72ESUXZveAU){fJyWg{srTx&MgN}>#ItDE~F=FEO!~0a@zUq73<9NJxfjfOf*3(r@EDi?@w_2GITBAdBIhEd2>vNrE$5G<Py3MIjRg~=D(7|5XsztYiIx*=&-dR{!dfhGUL01U)je;m8MwcJ!v@6wCE4ffS=Bf?FbAL)Lm>BD*N7cgz58^=u>&FTdFs@LDV>q~ab<u^>Qlr?UrP+UQ_nv))-Fpv1WoomXyGmgo$f?@CgRhz|0fa3PmAA)e(Y?9XT1Ljj5u9W=>^~*u-X6uBq(s4h`)rL)SBt1r<Dr~I;5{fAOGzszi_WL4vZ{d>YiLLwGK8P_!;mQel)$Q*c@E`-Def)8k+$rf$fpv0`soK_^F_wz^DodTfd!fB0hRzhP~7gjeiuekDGW2n&*p27E^pkwx4wFM?fFODi*KwwTHZMKS91cib!6CLL&)RAcRId>b%uk_zIw52Y_Pr>!+s@!6G;UO#z%Za>RF6YRlqG>6Wt~o$EreVKp-^RS{F(8M5ToR-O(bd4#%PBjJ0|j$YG~e{+X|nVJM{5O;%wtF|}LfjO|0@Z#O^v;n5YI)V}-vcbIkg{!I+2eC8Y3Ymyd>xf8uYQM2V6>X?OQ^I!H2?hy{|1^<lMvsKx}`(RRe%N}DGVzpeH8eo@{NUICaMutgdaf7Ojj>7ur<5A1j<kCa(x0QymVt!j~y?{Zn6NGJ<#MuIP09zkrpUdvl43@RUJUYOs$t!WrdG<>QUMXUNyUk>UlJE}D;hp?eRBt%^s#{JvM=<f0!eEmMRQg!G(WG;@Q&Zaa*^)WL1;U47AUP9<+(xAZQ}8iNnoB7k4aMD6eMyebi_Kw#rDG1hOgLqW*=V}x^USSME(<G2c6vH3#x?2R4?h?c;7}KE^`XC%=Uq&urq$Snau%wmNFTt@0`T2drz-rXdp6;S(dv{+ZmWe-;vPk^;xX?0n0zW8$v}BHTo%(#4Et1AgFS5ee9{=YP=iM<IWnutp9slPwcc`hRJuqOf#$$+f*2n6!8#%SSV~Y@xD)Kx(iF&Ihf;}DA`3qgi7Y**6KQy!NbE@Lpyx~?!=ESUc_cBCNK^GhnhHh}qlt0$oT7p({5b{%6THG0JdINi>^<}h&(L3Dgr3LYDGR-h@h@V_aTvvpM23whg&!Qlf5v$iX&7ai8tq7ov64&z{y)x|@=7q!5gN%j?>Rl1fdB8H!SdO{sIyR##@^X*((t7*8ZHc*mB4!(a)yS$ir_2wp~R>JOd94p!A2@TlSw5q2&%M%MJAod(6<n%Y?v8<a0Y=v#jI!~0so(XN@JqZXd*L8Q^T%!SvHYDNRJ^rS;<%;GsdCMR}X-Q@ZJHXybjC*q3^>x10O<&P%XB`0E?7ZWd`4obOQbl@R!Rnl1Ypp#K+`JMgTWx7FirAD;iC}|Je%eNQ|;FShp-qctYURSYianJnDZA2DpP3LBMikJTbzs2tmw>pbskWaS;}TzLl1JkERo&X*PcjfzeE26v6MW%IHX9bcEJPEaoU+^(aAhf`4o@n;6Zq70+@^868WEj?w0jB{0Vwd;xf~@x<sjLtO#gQJClijYmKv3m{9;M8<iuEX*mL*rBXR7KTg+I3iadJCcC^k8|*{foEZOqqLy~OtOIMS!0L7sK;oJWkoai3?jUb6<OGn;{u%-;Ih!ggbv*?c#l;OHyH@V(upxz5J^GC0F5)WeZ*!RgU^hxxxhbcOk)7<QHDISApkPVceMB(p!*mbmFx`uK$wsE`)3TmIL=TGN6X4!MJ9;Sh&A2;?-9lEhfQz?44P2_tjLOB)C^h1_+ssV1sGu@gim<~5CH;!MBg3o6&(4PT&o@MCxRT;hL^xJcd!wTiy}aYaUwhdGj;%ePw)knQyGUADI%(4f`kI*q-knnVvWZE<e3EPjaPzMjcC+p97a7#Q<1b7)|$~2F}ZR0&KQALEXz2M8-|%;myg5N8t0SaUk8xl*38I$CIHMd2Or!MFp@OGCqd^XfD$q?F2FqjLncZg`Nsqdc~oK#gWLp6Axq>*Q1%JfgE;Ptk318w@&qBt>?Yt#<7`lpv`@gdCW!b-fTvUVK1HJv8#|rCp{My+Iaf$wqYO0`&`PIpm?MmT`Gq)vQGG9^Q@BC1d|!$$W4kd%2mRq7)^N<3kl)6MjMLWlKZCe4LHk<l^fY!O5lUcQnmB4&pd+6mGSzg3O<HhA<f`cr&OijGMaG&QB`g_}EE-0f&I*dn2Z_uzJ;sQ!{4`WZlQ@7F%*$Xh>2VDa(#T!Yj4#W{0DHxsF<N*&LS(R+G*RJkSr5g7j9|Q?B1|Pi1k{Hja9RA}>{zshsbq+7_@D(wOFt>vEdmh0Y}3SWGm?;^sULrsSi|QG3>SX@rExY~T!Fcar1V{!2Bw?lj2CGmFNB$lWVnF9Cxe`Kgrk|*-)ZE%Bcq%}ipDUR5ylXt%m>^Te;6|pUxrpAJ2)#5^<XL^#L4hOybQiJB50Ic7-YkvDdOpZmZgCY;|~+t_^^Q!<IfnAV|W=%Wt2!T>y?+H@HWb2M^O`MjAm*11$R#aOU9oa3R_MCQ^p?>>0}wqg|TKrdtL|=8D*SOvVb)3<}~qVg~g<SJ>yT74TzV)WU?gLDdHrsX!>C*%RvD=I!%09l~=PUjAm(Tp`5`BVLI6{VzFalUjnPfAJ$rwK|k4X;t75k%m$fu0&#`+2kaVuRQ{C)hD|>x+{pC@o{c|a>UKk}JvPF&ftWOM?Xgk8ro~)<apTVpffu|KWozs)<4gCzyzys(b|xP<a__Nm+D1NjfPLc+KM3%l0|&<+w&8ed<l)H5*?RHTz{T-roTkX91AH8RvZ|br23C$g6N;dh24;>w49~=PfS=<}meBxFh-U>Rj*JVC15d}FapH5LN0`hw;g~-ccutR}*}fD*gy-})6SF1ff#>x22<<|_Z}B-j&PF{ZAPdjwab42J=QQ#25ufnDb9#ISp_mT_C`gYpsnMVEI39Z@m@MlT;=zXwXg<9~RzD#n5?%&VnaB{$ljk^iPEU+zf+s$wCq}8Ir2WY7CzwPfH#$70C&q}lNo0iQH1T+4N#Hp>!8f{sp;$CA!Os|?IbdOmi;ZH(U_pi${fN)VuppDlh(b{lj&i9Hj*Q}z1_hbaC{0^XJ$%ljgq$Vn!8T2eaXDEwhQ&<n;CzrD*Wfurj6Ulh{NOp0njq4lG=|v-HZM327G%<CE(r4;(|9V)aAXu^FqiZQO>j)Mg{h?3u}B<C@HvytGT~-S7Q$T8V>BE=Pw+XD-ofY00S3>RG?&RacH?s<ElDt+9~5LV?1&^#8J{!6=f{<kGaj%rOk@#UD?=x2VvYWn@ra!v-lQC;@sv$`KI0!Uu^^KXB8#X9b0IcAp&a9(S!SG2PZEDD$Ydr2LWx3{3_D*dLI##(h|`ZMSc)YXV)dhvtmAWLgvl-bmqt)$Fv^&tEQ9%sWW|YG^a%4Ycs)L6Ms{#2!=VPx8RGS61#zJuGcv(R8DATC&Wxt``G<oSo-?Cdob<N>lw?LTOd%r{44yNigk#F?f`SY`f6G-IMHNVv3F{I|p&&C#OkL6*e9nySko%nj4N5YM-TRyZ3o@e<oRNxe!*sIzq#_p=3Nl%y7!x!JpEFs;QRLFVb0#}No88ArD9B_-1!nN+KtYDi<-W*_&*;HPBbQTzf=pKMQMpT?B$H*z7rC|JIg=FvGS@f3b7oA4W>QTAL*s{Zi5BpjA$G405(+Y7Y&zoDiq9E5tTR%>%V08NS-$Hys=#w*jL5Y@lEzR&wSzMdJ{5S*j4_3vzg$p|8Jp0=6nxI?NQounbAf`)jx<wwiMhaYW(N};{I!OH3?1ARE{6q~9is#ZAN!ynL&x_C$r<oDL(D!UPrdk@5l46l+a0KJ<H`pi?qNY@#{}OZqBYEBoKt6s0Z@<`PqXC`vH})l#xoqrg%%HHGd@D1om@hA&ft-rI1OJxc+Sv?URf=8&WtmWQZhMs&d{kor3?n9GR~GtvWjt3A5KUpikhfGq;rDb8lE!~X}<bmDo~Ii86e{e5Ll3z7~wi&zKP*EGr^>Nskwyb%tV%x4?aJ5&P<GP<(6y=lbPUz%kL3BI>FG)*L=ckCMGz<ICS87B*oy*$(hu+Yw<ybhK7DB4-e`*%*8{R7sDJpXdwS|-W}}HQqp>9Ni>PNQSlFT5{d4oRIFBKilt-nUevC6w{)yguhiPR_$%JU+|oj=*s63ElE<5P$JW0}g_m9TRTe7k{ykFES}u23H6gd}$kF{K$as_1%t(R$lnX?jhJ_a9nd($s9R~hH{yQ9DB0ODs_{hk%$%SL(N;5`==0g1#b82<i2U@~tDLd<{_W%E!Y_b6-o2>6-1G6tKwA9S(+(ibXE9qV`Qv=~49ig)32osFdB)niw{WG}tRQh%A!5|8~9DJZoZx<h@A0HoQP|c{zis`B;1@?4+hU!tVZ7x6LERe9r1+X}JBY&y*sU8{9D1JhOYHX<_HDjUWaS4P4o2=j%=qV%e$@S-1^a4&rL6@GG<`|;hnL<yT`b^98byqxe%vYsqr9n;7DLpF%mh&dYtGVLBLNT98PD~hnC;fT*Ah+MrzW&L*eOS2Q!ZDcK94;dHfGE&%+l3-@m^aF`t9z2m7udDQyq#GkoWUOaUyrVopTv`p1OJ#CaB)_77w~Y4TRe+m>WyBUaeiuGf3?9Q9S=mh7gd3WQm_IH+L||bYAovxkH)pTdbA(%qp688rI_8%pldQtHl5KLxQ%*g9<k1fSXv?M_2R-zncidrD2lrn-ej`cd{}I3C{L+(3KwTZv05AV^<1-9Ccix6orWrT?Mdy|ZO+tNu6BP>bI?*FEz?DVnL2DNq+xjssUrjfWcuwH=}xmnM=T-S^y>{&lJZ?a$bNu0L8(u@<XT)rj^X|$PdBsG#iLg!MdzjXp)NQ|292gWTR8z(*eNa)8jZTVYANO^ycCt;*rd0gen9-gfdTdBgmsjkUHWXpB1j-J7hHsmJ!SXbV;*|qe3VT3{#ne4jkan`4dtzMpEsVh7~u?_`AG{ZBm%23y>;M>C7mO%3=TO_Zpp=y%?;Nu7UhpAVL@(qVYMwzw|1{Hi<8{cz)^WL)jZ_jyv)uZmT0Z~MX_DrC!`lUOpgwy(!=SrlS${`|Dzc=1{7QEJ>?2;j#g9;q8iIK?jOsY^k^=-BR8s_&v#aug`>q26(H3jQ}1?~fj$9#2De2F<z@0<h~@AgU!Y)zCkSzT6o)ms0^Kr?mHJyimW}1p8#mrvdvwXWdU5^2`R=W+11HfE2CL@n_Q-Q3yif98M6i$g3I48Bukzsk@?{xs3|pi3Q{4U|od(IsuVC1cgmbjh!1Lv9ikjO)Sun`MPx3^o&c?UjufO^DhbI@;SI@&ASi!q+$$PfUCaI_OTw7V8fHZ?n_raC5M_;LLDddh;$A7u@=pJvt0^49>T}pn9)u^a@VcENNSBswSY&?AmBV_%-09mA$&E-$Ll`r7YQA5gLKRk9Yo*M9$(x9_`;j#DZjqa7J-j(a#xu?*JxEi^ghb?wCo_`2^dS|~`zjqy*Y&?Ciw)!2WQ9t)p_x#n(_nvR>e)!k!^@rW%FE{>rcYF8xIe3pn*B*OUFT*g_9<6$B-`zZS2`YJSe+pxULf-A4mp4z}W$m}~_-R)jz>6QAeBwv8Ew+Ef5IcrA!vEGISsXpG3sK48FX`ma0-UJ2&aXQ)Hf@Se-mbO^m2xl$h9!gxP@5cfN;eM==>ExpC<DSxU%joidkHvKrR40bp(JVKM51e3JZvHLjz=lVUV}?)q%}sdhl=RP><n%$Uw{?E6t|DQ)Z!ocy^^qudn7i=LgvqbsEmWH6j{=0AfnUGHl=GiHQ{ArzbNjQkp)VAI0X0r-}cG<ooc)COQ7G)rPydZ7-d$EqkxzFSUcF^v1<Kz-hpH6d}VH)^Vft!krwhXb#N022LxFbkO(iTHh}52#{yr2BWhdmKUddd6n5EHfLyR&a2GzkwEpxy-`x(|+m2ejduzqJrPN$|^f|QmKDvrIfz^r|Prra#Y}+}o2fNgYKdZTj<lc99dj<D`%Y)O=W6^9pzvg}Tv3LF>@5^@>b)cPeWAofa?~QXC_n-09-1KVm^wYIRD=Gy108cEeP;4SGDk7lyBf){#2f+4=_wXzykZY@#*H(Zk{?&<P6HYo42S|L~z4RG;cJ0wJm8TPa7I)j-+nRg4AT|LV;ncd;S~5J9B{-yy{0YZ$S6zg%BD^uKbE66naV{1cMuYBXJQ(apjzWztLj{8Ia>7AnD3ejntPIn)uuBM{W)Q61kKgiESEwnbHTm?3_vA`MS`+oNM$k>EHQ}l38*Uw#-_Uf<8HBY}BO~+Z0|8hm;do_4V%g1PWjg8ONFacoHZl>`QRxtxPo}0{s99QVW<o&Xh4vz%f&N9Y{G*BCble#X0!az$11BO{OAL1#273`bD`SSi6s>*0zP59J1XKq%Ke_?CdF}b_?i&w+UD5DpnIr%v(M}BiGl8sBpG>iUT4FHTB^R@9CZ|<RPunNN!W;9IQ`)O@0XZOIE<BxBUlYN)7}HuA7|e(TWak>rkVa{((THpmOyMP)=&%~aCas3+mP4wKXw|b=eHLSDSwCO}9d;8B$2bWg?S%?x%kHSu@Cuy1Jrc-xS%BrdvSbMdP8h9xDss0iLK&RzjSlq@MekSph{olJRDvmnWW52ecA&B|#>3yf3k=%3_;~ZorS9F^-pB92o?^o2*&CQPt~3v<{nF;G<&7^$6b9M;@@n_;9hCS%GQE6fbM=Fb`=7%f-uZ_<<>%zRKs~6GU)}LK59pgv8`&5El*L-fwGkZUBVn)cGrv|o!>bQX=WOv@*r$BzBx_Ipno%e};ax0Dk#B3-JCnTDR^>K<oSZ!k2Uf^QuEGl~^;JlU>c2MY0TuhyJ9Edo|9JD-4Mec>-#MEf{H1&EGraZu?5xB#Ko=%6g3xDJXxs%1s^5Y$oLi1DUV$NuweiN|iDT|ko+(tFV>x{`lgCW5C<aeq|M3d}sxIzd>R6ei;+Wsu==WGKIyk$x$-4VlK9G5UwR%Zjp^qNrv1wbd`>l@D3JcLrqzO_R2FQrs#@C$jQ!^e?(pt(jJkaQppSZTFcNGs@Y*pysNhJ+T1wY~#$<k;e$kb6LsW2i3Px@0#ptYsM())slG3@}4D-~;4)PW)rJJ4!3G}pu%&qKECX=0p{?0eIc!bzJO>`yUm_|sU)N+6!Pmf*}M**;iRn$LiWB`O3FXXI%>iU>N;;N0y{Dm~!8Lp+hd&@@jeU`xGVt2YU~O#7Jyh8Bl%Hgp!*1Ud~#p`1J>bBOBI+RQ?b{K-yQn@op6*l(ypmz|3Zq-BNioTZoHw6l%BH;;+zE;wg<APWI&Mm-5B+9pW_7m;yLmkLOcz~Uzvcg%aA>NPcrp*7(dYYYyomI~%gY+Kof3KH6pEO3~R!|;Kt4l3b9j>CFJggV+|+^Xggw5{)B4<*U7oizBV#5p;HLL5k}#N>x^AR-P4p;Rg2xa5liSS(5juA=l*w8WNB@~!AcYqwz}3<E?mvCL5P&eZEw@o%q`mrU=t@%;nu${#FBH+xD;0YZcR%a?W*U|(-@nZ#bi9Bs``t%P}}SO|lbjA0-&k7^tP7Jp^^@&oU)(`zfAudVz}i-<)XAPGZ%0)vK;QnU^j6dO8u3Y|KJl1y0{8zO%>R00phVY!bFB^=@Nl^^P~XNSk3q~;z+gL8gpD2{f7Sqp_c-mJ6<^Pux2dx&9x>`D<;`uQREM9HlVS!kc|QxT-zA|`B=*7u;=&X!5Gy?ZX+^<*m{iM~Y_Vr#z`Gl|<fW-GuKqJBJv9}1~DXvBy(vUh*ZIl1MLoN}h7Fd6sB{@`&+ORX7(5CS&F@c$9)fXC`n%3^$PlBn|E$|R5!c%Q$ue)TL_la{?lU!pegy@%aP_g{#8=n{h%ks(KLsuh@et$sX`<7nIfDkqH3g^|G^uV7YTQdnE8HR3y*bZRDK-vUgAb#f8@5eDE<Gytb3a_Jvo0H&TY87G~|jZEZ5#&u(EYpE6G2R&6|KBMSrNZIn1db_(yuUCwDhzYZd9^w?D2ah02tt5Vx!&W>BO%57vY|W*O!Df$+u*EXtZfm|X^Z&JXEj)G<SNgBCjI6oCj)$tQ>gw(rjChy@B%83d*eJqiR9z2y(7bl;48-z|gt80Xh(~s{%aSN$LnII=VF}tqE1u!ccs$`>>{r#V>b`yN%-CbhMu5S$U-dY3&Z$#X=X__`kCEphSxidwr}`i}xObS}v~#lb!)zLl?`G7SbK&`jd_Cvch1z#^p|PLkj(dCMN}WdgZ0j)U-9ud_h|0TUQo4hm?SC^`>~3RuN1yVn+$+LuHzibGFAw09%k?eiQnwg550&+opEPceml`L6(h%r)@dqtjuC0A<=ePJzlP<zor3YFIy^HTO84nm!&(X@bj_#}O?f>Dko1edV@cUmL{PZ(_j^@(7_f$)O@HDI<d{&IA@qwvywn2Vs=~&sUlZB;fCDiSDr7ik;VqI=^GMyYEmfuOCtQrjF+bmvKc0Iy>b-OP}2<K#V-SyNVt;wy?)F*Amxpw9+c30PQD__sb^+(G)?qjSYP*!`-71?Zq;$bga;y4_q>Bnw!u)Py3qr0t~)VY(ZCe^u=-Fr=5<+IIg*QVf<roO%0Ij$kS<C><4{<DBvYMZ&kETO;6x0)j7*8lwW);m7~VRjs1dz<GsI{5I9&Wqa}GXM7d{a0Su|M<VY`Q7XNL?a!4*`sF|ai5)Wt=Gp&FmkQR)B4ET>Xi+A>s8b|58dSx%mFg?4oR2hqyzB@jGf2AZD$1;`-N%?g*jb@qg=m$Hfcw<1=HZ=&GS1*!QpdrM=kDgm5HOg@R>MDXyxx_CXP9oFpipq<GYhF!@81JIw>a($s*ASKbh<}<>X+ZhtiT9mg!=k@{`PWag3JUS!#@E<Ya2`xG4FqO;tzm`KX`bwn<0PZF}#OD@;bwN^$R{F!w<7v-i%NYznuZJmBgHCJzXDH!S1#6!E`&Z>7{MwJw>)nJ*=|33R;uyZ{~9sj>r<72R2efH31<P||Ir4e)G>!3@#?xLc#2EOrx9R<%5NVz~S>!Sw4ERhary`FLGRo;(_0|4i)vx{fdf!2CMy9h`q<PFBKH$22(<;JJq7-7zIgtMIYi$<F4w*Q)bv!2##mx%SqunI)#3=0gAFgRlPW;EQ+WxXc~j90{Y$yp}*Sn3zi8W!4L$b&0n<FPBxT9-9J}A{O`4P7S4_a{GG8{Z1h!_p^BLB<(n}6pNQSlab)FCn35tHQfxRb*wf9*LKSWlwFvjRms^$CKMWWL<)7ka~+&9nDeKgnj<EQgXfiPm|WK^t!9yq$A#PMamkymc%<^7w?kLD@NvKF+Bq?({r<3c=d&1Ssem23{5k4$aQKVn_du;nGwUpebtxKM-B^m$#$sm@X^IR!xa#cjs~gKNu9}ye%Wc&0<i5Jm?liCMqR>G{kA{~5&g#Bf>D2O3t3<;s!(g-b;d;0CM~^)HM07!JZeZTnc(~c&WoJlsCT0<(<2W5Bse2Y;W9)zO>cOu*-T%X9?nv>^|J*o3AAIra<|e+G+V3sP<q=KC1VzW){oc*V5%x$s0Wd$@DszgpnmX(-nBWDYcG~m)0qyMfr0{{?R%*&~XQ|Iu(F%h>5#H7bK6R#hc-e$FltCgce&OTSZ@u-Wnlxd1TqIk!Jz#~hJ!r<DTlx`*KWlQRZHG>CS10`*1~F$Zl|i?o1%E15?yLOe<3AsK^>Nt6vfhQE1yYYto7Jveb*J4J3{|Mzbem=aU+L9tw<kTCDtSN{s!v|nT-_LT3+xg)*u?V8fsWsvu#hr}ttTu#iedlt|BTLWURlEBt1fKdi}!DS`r7`DcW-|ByKnyKL)--Mv8I0N>A(G9bny8H`~UKLcbxV1`}<$Ned}lczW>H2`|o^s>rLGM-~FF|@`KA8Snl2Y;_HKVe&cxIl|MEw@BjBN_J8^zUXCza(f*e&@4xpa_s|d0<qa%?4nBVS)*s)R`I!$ayfXde!$YSx_>;k287E>)*t9fH?HJFIY=`^cfidc^x9Zw)g?Ex@cJLj)?D)etwYjame7fDG&UW1X8*d%F_R(LyzH#e|_x5jmUXrRL^v%ydIQWm>G(B$;`)*&EbWYM-;nBz^|5{lbM=QD=)RpO-Kgg96kOl7&@4tO+Vckq7@=Divt>VqrxTtP4*(J)}){(Wh(bqCNP1)Bw?PZlW**7u#&bQ$xB)NG0q$n~mI~Baf$l;3wQtU3*%!_92?B!GE9)I-NC*0m$nKnbN1y2{8a4pmdwLG$|xojD#)_Pap<_=zf-L-J~$7fEjpE<2pPCvPF`ohZTOABpC?n+NNc}>%{4kau$q;O{#^?OxHmio<5$8VOfL{M{+prj61P7v#S(X4O2Xxv01ch_2h3bgmOe`C~7!y<K<-epGn7nUCYOP{&!$g$`0pgaGVj{o7h+yT+c7ovZRu6sg{9~Az=>FK@kbz6lc;-_1P?L}vz>lGjDxseJA^t^s{O47r3%AxFzd-!~NbEi(IoJJQOoeG{mX8%iV_^p$mC;lf%00;e1ibix*Zs@fhzjO)Xj@TwUm^ilwd%f@=7Wro5A>aG)kMYl$3m49uI~P6q<am93Jee$Cx^!vp!3WB>K#&K0b04aR6g`;XVR_3vRM#ZR@-%z!0e>;B?llDL-S^b2?t!Rv@ilGh!WX(kE=f48Au39>)<AunedmGle9Y+``bp3<isLbjeaQWh0_B6|AIr01z|wXo7j8erI|j^(tDnE`g8x<n0J;n1<<XBYLx44R{J6c=IP26Wu#a50c)kvd_PN@=de^x-UOXQK4%l-s!zyes7h!E>d44^*^eA(^+R){Z`C(tX$*KpORpVsUJ3KtDF+nP*tl;d)W3MeWFc2kFR{df9*)+11v)`+Sv$Ny{FNM%qb?I?aZ2)G~9NpEBR_%H!oa`UlHqHe^*1_mqq(oFuMszICFhp7kH4K1Tfzj6sDr@N0`Wd)W)rzSiPsNQX#IbR&O}G_}eC&bT27{ZmD8Stsr1KaW`X@#Qv86uiohxJ8Y;AAqUA0=QuC4A|8#<w_T3x$DRF{jLhZQ_th=(_8{`c!gw_+K2!_aLjf^oxj>cE-*@u1wM8Exv$(tISe&h@P0o6<KZl;2)E>z<sge;flkZsxQ3Xma7@{B~DdH=4q^>s8>W<z_bzuT=BG1ChOa%5muRhE4ZsU0Gt!^*8^(shcmY7`eT6ZN<AQO{VuMr_Afdtt+_ac_-X+Z*VK`c)}xMOz(KYE2mB}<rvs(6wR8Wj)^Jiw)8^~&T=bh5acO*wd(+K^Xp$@5nZ~b-+K4P{*70$<iGjZYy1EDrrZ8``O70gaWv3TUMDZLcu{_aSPBuM+Bu_xB;-t;3i_*=HKv^vhB~+8w`+oPgQ}P*uj9InZbP#EJ@ihsI%<CR-U<5Rw#nP8DgD-<J<W7{y|12t%6%Tc;4(&sk<0bYSKGAJiC>isY<-2Ru<l25b@672IoGx_SgVIk*zg^iM5m_+Ivrv-w8ZQ;2^!>jpQL*$)vew27H4a(#KddRrl$7lN`x!7i`ro-Z_6x}SXRBwwy<9n!$c27bRZs~^c-{U_3}k0ww>g$JDbsUS8~@WzOy&bisqx}&25S2t#@9&`Q<Mh<dB!sql;u9f`<#&tmYj{z@~-guT&_G$l!&`n>mb`c845%IQRBB*A4J(v9BXYAGfl`!tq!DurFPBm#0%bpFQ`SzxVu7`Rw^2ze8VU1CSYcckH@`#^xS-nBk~b;Zx;h_v6aJa9s9a+>{mVT$K(1d3|el$22Ck+TGsv`FlI{YvTeg+~ua*0bH*D>tH2cf!Q({7CyJqAJwm{G+%{u@xSJ{$UIYh9lf-=wuV9#gaE_pjRG2t3Ug(3!<D({MvW|o#!^M0f%=1YYMKo8dUmraW|%HJy1zRlJuA&oq&3k%^_}lb*ajOX+#e7AwS;YatHB!q<Y2Wr>Ov3RS#Zap822X@<FVwIu6`YiA{>J*93|}Io`-x4#C3F|Ku$dV!~HuITzxVT9$~ARl-_|UFf+n&^4x;Yn(afFpZ6h*e{;#TYka+I-rf)Vy^Cdf?CVt}doAK?FMgx}v<_oVOdWGF={R{gW?6Br6Zg)y%e%&{fA89LN3LDsUbuFN?_In8f4g?g_OiO~_^w^|-DTgd8P(oX_pa_SNJobr4GDQ3#PK1B#D0!_`tcw2vkrwCsScv3-L=|BP1H#EX@xXV1M=1=$Nug)(?qG%n!B@ITy1_21!c9zKW-tiv$+**+BVp&I*vGh-o1xea#gwM<=P~=YPQW_oYlJ7kuHCTPuFq^rliYD<z|pF0n77y#|y|B*wH@9j`fV~>Dfn3wMz6p`c1@?SgHqd9PNMl!T!6i?*IJV25EP0l+(0gEN#6>b7a?Dn75g4>Yx1$!KQUix3(88qxEX5Lv*GYYcxF-1m|ELf~*1JX&>~WF0tB-97bkI9fxoG%l8g`_3^EDZ?rjehBw`P*)<=Z?XvLbh$k0wPABFE!BpWK#jdr}L3uS$@k;%;!cMLbkz1-S4v@XB;0df;Dg^VP;>#nzl{Ty`9Ioa1_KZ<I1Vp*-qWIETG8KDt<?2~#MDKP5v^><n`vYcMT3l*cHi)U~cOOFYWqROy*wIz%&#g`r^mrdk(6Qo79$1`LO^B)8E~n-MGu;<GTP6-W^vwpS*+X9%?MO8PCAqzRY|}4ZHvHaG`CU{=Fm%mk<e<M_G~EO<Gd4*;XVsQAa>6+3aqeORl({S?Zg58J;4B`I9FC74dFH7j5+noyyT8H$^C#-jrgwoW&({~nvp`y=fWJD1Q(sKVzZ)&J6cjcn^iC<4<SvPSh~Q(j=4{Mf4yu(F(YL3O7l%S6gW@W5l`2^4zUZ49Z|uMK#=%D)mDnt^;(w-?48wYT?+|b-a8nn=ZEg~{PR!~v+s5tw)T?sWXH)My9JwTi4;GlFWw`w1Q`K!}zpPFiKU_ZbeDAP$rP&SWrVv}TUJq0yEJhDa9|L(9$?AnEyvt6};p9{OJ*vUFwt9$=N2FZ599=5?I@}cRhUAz<w1o2M23DFKkJ1UtZ+Qf}A2kG6Ei#@7+D_h17hL#fJGC71^AnKI-7W2pVva7Mfk<_9e=99@o2I2`7(BRse&jI-_i6&ifbRn~z7N>Aw*eb_OQ&Ga#fdmIzJBWB<By!X@c8n&25VAdRmN<ZGOLA+wc<7xEU_vkR&q&s9Oss3q72VyTF_ibo=_oelBq0H9Q<B-A~p$>N|raQAd2%i&#a<~@?6@&a+T$|utgs8IF*J|!v)WaoN*<2p5!?#(kzLKEJ@mO&5JvmYj?SH|NpY+B2HuUO%znABDPdm#d2$vN>f9nOjDI9t&Ol8Rr6RRSym8GFQJ)cF^va{uCZ^*Dx`^1{3DZ0(}d)OLhA}8G)XO|sUcPri6DhFxi$r1Nml3#jV`3deL=I~Dz?oPn0?#Vs&1r8raCqRgRX_hbH)mq$yBMF$-)?=6P;=+B`bKA@=T=G7K~+rvX~Uc4%ac+GO8>;PkF&|v``Tt3o1Ytlc&0n1v5-(NpoccQ7RLODrk~QW~5~pq?8z`jqI0S1MQU<e!dD&hl8+!nS|wuEE1)`cT{CD#UC)E!OvKLdmZIiMhw~^3n?{v#VKp6SeILOje%U*4dfFcR-4RZgvO=-U*wV|HYPd-k*J0UWrd=t1w9Ih_GkE`RaV3j?8VbKZ;LJ62Jc#{tKusZ?2==o1POVfER+Bk&t#S`QV0t%PfRYQiVf$5p#=wv2&R;_3B&;H;Up8qjKaIDuT@$UAQ8StSFBOFO|3!E8Okj*C7h95#HmgT9%pGv2t%u4Wk_7)MM~OA>Rpqv+%$#OGFlU!a-rkGaH+s^(tv4HkznajXs(k)QwFBunrN2gcrUl8H?;vfR*ZPpC|{Rf{aqFhO_OF?P^$!^al%;^o17@hEMx`TVvG@4isl#8X!HtnF}XnTnrDjXbY}5A#Y1C7s3n#p2~mt`%QO=V5<^mz>RbwoKPjV`O$BqD39>^Ea*R2*hJ@8rOYXZ!r@Dp~ORUs9g<&!;gp4J%+Lb|Mf(c6t47!#@Dq@K`xr9I&!JzFxSC*vQSfSFG&8?zKFTzQj+B8og0EDQ*GD(bq4we=_EyN{_1&eJVGW6B(g6K@8B9F5y6Pl6vRkg|#vk6mRbt1Tc!X#ScX<o!C#t<nk62?+3xQ&%&38#*+X~vAjC~?ieIH;%0YWl$6?V4Tl2rL@&_jNsGs@O_Qvdrd~+d%T%KoDpFMj!<z=py8v(=1P|09RQ7k=77<jGE0r;)J8RprT&7#1k0zs$h3Wxv~|sIa4TE#5TtdX)yZeH3nZO%XNx|DUxB>In6OMD4xZboVn8*JR^llL{PLlB`Uq5T4cDk9!?l2tA(P3<R~Z+DUBJX#f)W`OjV-L2*FiilY+%eq(ZTjaHBJBtq`WI=O5f4tVlNsGwOoc!4%1Jk>;FGfxi`t6PcS##i&o2LMKI@fNVCVf?(c6*GN{d#H1Nb^0uzYsyp6aDXZ*MkqezFm;n?*A6tT6!HjTfL&XVovoK>Y0g@u4VA>+jFi$hE9oisB>Z>S^8ScVRR|R*^FDj8^ptYeUhOq+!!fC~v!5+#eCyBsV#|0EuoF#>UMF3SmAodh9Y$DM=7PqAj9IRATPzwde1+1%_3upzSQZP5oF}p!%VFtrMlhC+^m>5ivKu1!T7otd+A(?Hf*lD5W+Fa6QCBm_dp)vGoDxqLmmSJk+(!kKmVM#kXA*Q&XA}b`u$cZ;72-}-xnC(!BKkVPG%u;30#tc&}Wg<x=htWnsRWy^JeCP@e+k;4a4_QHh3dDy>1j{lEcI;+|xtZMEQou@-rE-;H@`t(;&<jMTLW1M4Mt~;OC_9Z|7aK+6f(vTEnu?*-m@eX`iVn4NSh6}LG*cKWr;&MPAU!;V2s6c50_GMHmZ*jH1}90F2QWi3H+3tTr!zCOo&X=t$ndIQ6l-Wg$rTKIL!A{4{gcF;=P*LRd>PD$Tv<@8pd{zHl^IVHhHDP~oQWKa@+|Bs*!wia=a@!Jky2=wGzKA*%%s9_!6GqSgXkG#)?#+y45A4$p3a0CwJ|ykqmvj|PdNi^6#;`u&MZ`zf$*Rb4x0iDkuoAvrYL55BJd<rHkpM=>G+Fd#lR?$xg;dk3}%PSbFeT4V?)hmm|mqJyohybQdDbUa#?7^1k^qpp2%YAb7KcKq*z!M9I~k)<qZ5Ti-JIE3YBS0^jLUtZAmPlsxU8Qv|u8R=V5n%!=Ti$j)mrM)>4us&?>o+C?6JtO>!cgg_X$E<ltw5d4wihODKLRG@k`?kmSy0r6P&JV~iv@jCX}~BD8t}ODZ)?y9qTdFrmPNQUw;zMQpHCiDBQWY*xS|=2?sfgY81H+@!D|pc_m=powBAY}haqY5=P$yh4b;)aN)lhpL2HDZIHc9GF6#6l((U!&0mOd9fh02Bjs&Vo3>UgcXRj8&&|cfK|bSTY9O?2ra?YgYJ}tD+KFf^j^RoO*OR!ObQi;MS;fjg=sOvybSG-3WGtzvQIhj6j=sIEoM43WhQ5_hA1$I8w??dSqnqNvHC7Ra2T)|#|i{XZHz&YLtkUQ2G>DQ6om?yf838%3xNctCU(jUOGhXMn}WQUcVn)hrm)K3nUrp=gLNmk+Ok}uC#aUDdT@VFgIg)Xio>!Pssi>WY)}&$>MaLRXoUfE>ktbejv0uGmS(WMgqz=Vd}?p+{{XCod}0"
SOURCE_PAYLOAD_SHA256 = "0238a5e39c68d0625decc976abd7b27e15216293179e1132d5fa2ee5c1f814ee"
VERSION = "SOURCE_K234_5M_4S_60M_20261019_V1"
METHODS = ("UAGMC_SOURCE_L", "UAGMC_OD_FOCAL", "AC_ICM", "S_EFFECT_EVENT", "SA_EFFECT_ICM")
TRAIN_SEEDS = (601, 602, 603, 604)
RULES = ("SOURCE_NEAREST", "SOURCE_QUEUE_AWARE", "SOURCE_QUEUE_ONLY", "STTF_NATIVE", "QTTI_NATIVE", "CAPACITY_RATIO")
CANDIDATES = {2: (0, 1), 3: (0, 1, 3), 4: (0, 1, 3, 4)}
NEW_COORDS = {3: (0, 10), 4: (10, 0)}
N_FRAMES = 6
HORIZONS = (0., 5., 15., 30., 60.)
SOURCE_PPO = dict(n_envs=5, n_steps=4096, batch_size=512, n_epochs=10,
                  learning_rate=5e-4, gamma=.99, gae_lambda=.95, clip_range=.25,
                  ent_coef=.01, vf_coef=.5, max_grad_norm=.5)
SCRIPT = Path(__file__).resolve()
_SOURCE_K = None
_SOURCE_PID = None


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


def activate_source(root, k):
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


# 学习库只在正式worker内导入，父调度器不会初始化CUDA。
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


def make_resume_hook(directory):
    directory=Path(directory)
    def hook(model):
        pair=save_pair(model,directory/"resume",model.num_timesteps)
        save_json(directory/"RESUME_POINTER.json",pair)
        save_json(directory/"LIVE.json",dict(phase="train",steps=model.num_timesteps,time=now(),pid=os.getpid()))
        # 只清理本run明确创建的较旧resume文件；保留最近两次完整更新。
        markers=sorted((directory/"resume").glob("step_*.json"))
        for marker in markers[:-2]:
            record=read_json(marker)
            for name in (record["model"],record["vecnorm"],marker.name):
                path=(marker.parent/name).resolve()
                if path.parent==marker.parent.resolve() and path.exists():
                    path.unlink()
    return hook


def setup_worker(task,root,directory):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    os.chdir(directory)
    install_learning_classes()
    activate_source(root,task["k"])
    from loguru import logger
    logger.remove()
    logging.disable(logging.CRITICAL)
    def interrupted(signum,frame):
        raise KeyboardInterrupt("本worker收到停止信号")
    signal.signal(signal.SIGTERM,interrupted)
    return directory


def train_worker(task,root,config):
    directory=setup_worker(task,root,Path(root)/"cells"/task["id"])
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    start=time.time()
    set_random_seed(task["seed"])
    device=config["device"]
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("请求CUDA但当前环境不可用，不静默切换CPU")
        print("CUDA:",torch.cuda.get_device_name(0),flush=True)
    n_envs=config["n_envs"]
    factories=[EnvFactory(root,task["k"],config["train_horizon"],config["trace"],directory,task["seed"],rank) for rank in range(n_envs)]
    vector=SubprocVecEnv(factories,start_method="spawn") if config["vec_env"]=="subproc" else DummyVecEnv(factories)
    pointer=read_json(directory/"RESUME_POINTER.json") if config["resume"] else None
    vec=None
    try:
        if pointer:
            vec=SourceVecNormalize.load(pointer["vecnorm"],vector)
            vec.training=True
            vec.norm_reward=True
            model=SourcePPO.load(pointer["model"],env=vec,device=device,force_reset=True)
            if model.num_timesteps!=pointer["step"]:
                raise RuntimeError("恢复模型步数与归一化配对记录不一致")
        else:
            vec=SourceVecNormalize(vector,norm_obs=True,norm_reward=True,clip_obs=10.,clip_reward=10.,gamma=.99)
            kwargs={k:v for k,v in SOURCE_PPO.items() if k not in ("n_envs","learning_rate")}
            model=SourcePPO("MlpPolicy",vec,learning_rate=lambda progress:5e-4*progress,
                policy_kwargs=dict(features_extractor_class=SourceFeatures,
                    features_extractor_kwargs=dict(k=task["k"],method=task["method"]),
                    net_arch=dict(pi=[256,256],vf=[256,256])),
                seed=task["seed"],device=device,verbose=1,**kwargs)
        # 验证真实计算图存在LSTM，而不是只检查配置字符串。
        if not any(isinstance(m,torch.nn.LSTM) for m in model.policy.modules()):
            raise RuntimeError("时序网络未真实注册，拒绝把Flatten MLP当UAGMC")
        source_features=model.policy.features_extractor
        save_json(directory/"MODEL_MANIFEST.json",dict(task=task,total_parameters=sum(p.numel() for p in model.policy.parameters()),
            has_lstm=True,source_commit=source_snapshot()["commit"],feature_class=type(source_features).__name__,
            source_reward="作者Scenario.step原样",native_train_horizon=config["train_horizon"],
            clock="一分钟一次原wrapper step",extra_modules=dict(focal=source_features.use_focal,s=source_features.use_s,ac=source_features.use_ac),
            source_fidelity="源码环境不修复；只接通TemporalLSTMExtractor及其六帧输入",
            resumed_episode_state_exact=False,resume_note="恢复模型/优化器/归一化；原仿真episode重新reset，不声称轨迹逐位相同"))
        model.set_logger(configure(str(directory/"training"),["stdout","csv","tensorboard"]))
        model._save_hook=make_resume_hook(directory)
        remaining=config["actual_steps"]-model.num_timesteps
        print(f"[TRAIN] {task['id']} 当前={model.num_timesteps:,} 目标={config['actual_steps']:,}",flush=True)
        if remaining>0:
            model.learn(total_timesteps=remaining,reset_num_timesteps=False,callback=CheckpointWriter(directory,config["checkpoint_every"]),log_interval=1)
        final=save_pair(model,directory/"checkpoints",model.num_timesteps)
        save_json(directory/"TRAIN_END.json",dict(task=task,steps=model.num_timesteps,final=final,wall_seconds=time.time()-start,
            requested_steps=config["requested_steps"],time=now(),status="TRAINED"))
    finally:
        if vec is not None:
            vec.close()
        else:
            vector.close()


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


def rule_action(env,name,rng):
    """SOURCE前缀调用作者原规则；NATIVE后缀明确为同源物理的在线移植。"""
    s=env.scenario
    candidates=list(CANDIDATES[env.k])
    if not s.waiting_decisions:
        return 0
    pid=s.waiting_decisions[0]
    if name=="SOURCE_NEAREST":
        from policy.nearest_vertiport_policy import NearestVertiportPolicy
        vid=NearestVertiportPolicy(candidates,2)._select_nearest_vertiport(s,pid)
        return candidates.index(int(vid))
    if name=="SOURCE_QUEUE_AWARE":
        from policy.queue_aware_vertiport_policy import QueueAwareVertiportPolicy
        vid=QueueAwareVertiportPolicy(candidates,2)._select_vertiport(s,pid)
        return candidates.index(int(vid))
    if name=="SOURCE_QUEUE_ONLY":
        from policy.simple_queue_policy import QueueOnlyVertiportPolicy
        return candidates.index(int(QueueOnlyVertiportPolicy(candidates,2)._select_vertiport(s)))
    slots=s.vertiports.vertiport_evtol_capacity
    if name=="CAPACITY_RATIO":
        return rng.choices(list(range(env.k)),weights=[slots[str(v)] for v in candidates])[0]
    scores=[]
    person=s.persons.persons[pid]
    for vid in candidates:
        v=s.vertiports.vertiport_list[str(vid)]
        access=s.vehicles.estimate_travel_time(person.origin_position,v.vertiport_position)
        aircraft=s.vertiports.evtols_at_vertiport.get(str(vid),[])
        if not aircraft:
            scores.append(float("inf"))
            continue
        choices=[]
        for e in aircraft:
            remaining=max(0.,e.spec.battery_capacity_kwh-e.battery_kwh)/max(e.spec.charge_rate_kwh_per_min,1e-8)
            flight=s.vertiports._calculate_distance(v.vertiport_position,s.vertiports.vertiport_list["2"].vertiport_position)/e.spec.max_speed*60.
            queue=len(v.person_list)
            if name=="STTF_NATIVE":
                wait=max(0.,remaining-access)+queue/max(1,e.spec.capacity)
            else:
                arrivals=sum(str(getattr(p,"origin_vertiport_id",""))==str(vid) and getattr(p,"sub_state",None)=="to_vertiport"
                             and max(0.,float(p.current_timer or 0.))<=access for p in s.persons.persons.values())
                # 只使用当前队列、在途承诺及源补机的平均充电周期，不读取未来请求。
                cycle=.8*e.spec.battery_capacity_kwh/max(e.spec.charge_rate_kwh_per_min,1e-8)
                wait=max(0.,remaining-access)+(queue+arrivals)*cycle/max(1,e.spec.capacity*slots[str(vid)])
            choices.append(access+flight+wait)
        scores.append(min(choices))
    return min(range(env.k),key=lambda i:(scores[i],i))


def replay_once(model,norm,root,task,config,seed,deterministic=True,rule=None,probe=False):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    env=SourceEnv(root,task["k"],config["eval_horizon"],config["trace"])
    obs,_=env.reset(seed=seed)
    counts=[0]*task["k"]
    area=0.
    traces=[]
    rng=random.Random(seed)
    probe_orig=[]
    probe_changed=[]
    while True:
        waiting=list(env.scenario.waiting_decisions)
        if rule:
            action=rule_action(env,rule,rng)
        else:
            normalized=norm.normalize_obs(obs.copy())
            action,_=model.predict(normalized,deterministic=deterministic)
            action=int(action)
            if probe and waiting and len(probe_orig)<128:
                # 只扰动当前合法OD副本，绝不读取尚未出现的乘客。
                other=obs.copy()
                other[env.base_dim:env.base_dim+4]=other[env.base_dim:env.base_dim+4][[1,0,3,2]]
                other[env.base_dim-env.frame:env.base_dim-env.frame+4]=other[env.base_dim:env.base_dim+4]
                with torch.no_grad():
                    a=model.policy.get_distribution(torch.as_tensor(normalized[None],device=model.device)).distribution.probs[0].cpu().numpy()
                    b=model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(other)[None],device=model.device)).distribution.probs[0].cpu().numpy()
                probe_orig.append(a)
                probe_changed.append(b)
        if waiting:
            counts[action]+=1
            traces.append(dict(time=env.scenario.time,passenger=waiting[0],airport=CANDIDATES[task["k"]][action],
                queues=[len(env.scenario.vertiports.vertiport_list[str(v)].person_list) for v in CANDIDATES[task["k"]]]))
        finished=set(env.scenario.finished_ids)
        area+=sum(pid not in finished for pid in env.scenario.persons.persons)
        obs,reward,terminated,truncated,info=env.step(action)
        if terminated or truncated:
            break
    result=episode_metrics(env,counts,area)
    result.update(eval_seed=seed,mode=rule or ("DET" if deterministic else "STO"),routing=traces)
    if probe_orig:
        a,b=np.asarray(probe_orig),np.asarray(probe_changed)
        result.update(OD_coordinate_swap_TV=float(np.abs(a-b).sum(1).mean()/2),
            OD_coordinate_swap_flip=float((a.argmax(1)!=b.argmax(1)).mean()))
    env.close()
    return result


def eval_worker(task,root,config):
    directory=setup_worker(task,root,Path(root)/"cells"/task["id"])
    start=time.time()
    checkpoints=[]
    for marker in sorted((directory/"checkpoints").glob("step_*.json")):
        record=read_json(marker)
        checkpoints.append(dict(record,model=str(marker.parent/record["model"]),vecnorm=str(marker.parent/record["vecnorm"])))
    if not checkpoints:
        raise RuntimeError("本cell没有完整配对检查点")
    rows=[]
    for record in checkpoints:
        step=record["step"]
        path=directory/"evaluation"/f"step_{step:09d}.json"
        if config["resume"] and path.is_file():
            rows.append(read_json(path)["summary"])
            continue
        model=SourcePPO.load(record["model"],device=config["eval_device"])
        # VecNormalize只恢复统计量；不建立自动reset的评估vector，避免终态丢失。
        import pickle
        with open(record["vecnorm"],"rb") as f:
            norm=pickle.load(f)
        norm.training=False
        norm.norm_reward=False
        episodes=[]
        for seed in config["eval_seeds"]:
            episodes.append(replay_once(model,norm,root,task,config,seed,True,probe=step==checkpoints[-1]["step"]))
            if step%100000==0 or step==checkpoints[-1]["step"]:
                episodes.append(replay_once(model,norm,root,task,config,seed,False))
        det=[e for e in episodes if e["mode"]=="DET"]
        sto=[e for e in episodes if e["mode"]=="STO"]
        complete=all(e["full"] for e in det)
        summary=dict(step=step,full=complete,ATT=mean([e["ATT"] for e in det]) if complete else None,
            completion=mean([e["completion"] for e in det]),ATT_completed_only=mean([e["ATT_completed_only"] for e in det]),
            source_ATT_completed_only=mean([e["source_ATT_completed_only"] for e in det]),
            restricted_person_time=mean([e["restricted_person_time_per_generated"] for e in det]),
            AWT=mean([e["AWT"] for e in det]) if complete else None,
            STO_ATT=mean([e["ATT"] for e in sto]) if sto and all(e["full"] for e in sto) else None,
            STO_completed_only=mean([e["ATT_completed_only"] for e in sto]),
            STO_completion=mean([e["completion"] for e in sto]),
            action_counts=[sum(e["action_counts"][i] for e in det) for i in range(task["k"])],
            effective_airports=mean([e["effective_airports"] for e in det]),
            OD_swap_TV=mean([e.get("OD_coordinate_swap_TV") for e in det]),
            OD_swap_flip=mean([e.get("OD_coordinate_swap_flip") for e in det]))
        save_json(path,dict(summary=summary,episodes=episodes,time=now()))
        rows.append(summary)
        save_csv(directory/"curve.csv",rows)
        print(f"[EVAL] {task['id']} {step:,} ATT={summary['ATT']} 已完成均值={summary['ATT_completed_only']:.3f} completion={summary['completion']:.3%}",flush=True)
        del model
    final=rows[-1]
    late=[r for r in rows if r["step"]>=.8*config["actual_steps"]]
    late_full=bool(late) and all(r["full"] for r in late)
    late_source=[r["ATT_completed_only"] for r in late]
    def slope(records,key):
        valid_rows=[r for r in records if r[key] is not None]
        if len(valid_rows)<2:
            return None
        x=[r["step"]/1e6 for r in valid_rows]
        y=[r[key] for r in valid_rows]
        xm,ym=statistics.mean(x),statistics.mean(y)
        return sum((a-xm)*(b-ym) for a,b in zip(x,y))/max(1e-12,sum((a-xm)**2 for a in x))
    valid=[r for r in rows if r["ATT"] is not None]
    best=min(valid,key=lambda r:r["ATT"]) if valid else None
    result=dict(task=task,status="COMPLETE_EVAL" if late_full else "EVALUATED_WITH_CENSORING",
        LWATT=mean([r["ATT"] for r in late]) if late_full else None,Final=final["ATT"],
        Final_completed_only=final["ATT_completed_only"],
        Final_STO_completed_only=final["STO_completed_only"],
        LateSD=sd([r["ATT"] for r in late]) if late_full else None,
        LateSlope_per_million=slope(late,"ATT") if late_full else None,
        late_completed_only_ATT=mean(late_source),late_completed_only_SD=sd(late_source),
        late_completed_only_slope=slope(late,"ATT_completed_only"),
        late_completion=mean([r["completion"] for r in late]),Final_completion=final["completion"],
        Best=best["ATT"] if best else None,Best_step=best["step"] if best else None,
        Best_to_Late=(mean([r["ATT"] for r in late])-best["ATT"]) if late_full and best else None,
        final_action_counts=final["action_counts"],late_points=len(late),checkpoint_count=len(rows),
        eval_wall_seconds=time.time()-start,metric_warning="未完成全体ATT为NA；completed_only与受限人时不是全体ATT",
        evaluation_trace_role="作者训练trace回放，不声称held-out或final test",time=now())
    save_json(directory/"CELL_SUMMARY.json",result)
    save_json(directory/"EVAL_END.json",dict(time=now(),status=result["status"]))


def rules_worker(task,root,config):
    directory=setup_worker(task,root,Path(root)/"rules"/("K"+str(task["k"])))
    rows=[]
    for rule in RULES:
        path=directory/(rule+".json")
        if config["resume"] and path.exists():
            rows.append(read_json(path)["summary"])
            continue
        episodes=[replay_once(None,None,root,task,config,seed,rule=rule) for seed in config["eval_seeds"]]
        full=all(e["full"] for e in episodes)
        row=dict(k=task["k"],rule=rule,full=full,ATT=mean([e["ATT"] for e in episodes]) if full else None,
            completion=mean([e["completion"] for e in episodes]),
            ATT_completed_only=mean([e["ATT_completed_only"] for e in episodes]),
            restricted_person_time=mean([e["restricted_person_time_per_generated"] for e in episodes]))
        save_json(path,dict(summary=row,episodes=episodes,time=now()))
        rows.append(row)
        print("[RULE]",row,flush=True)
    save_csv(directory/"rules.csv",rows)
    save_json(directory/"RULES_END.json",dict(time=now()))


def collect_summary(root,tasks):
    root=Path(root)
    rows=[]
    trained=evaluated=failed=0
    for task in tasks:
        directory=root/"cells"/task["id"]
        trained+=int((directory/"TRAIN_END.json").is_file())
        evaluated+=int((directory/"EVAL_END.json").is_file())
        failed+=int((directory/"FAILED_TRAIN.json").is_file() or (directory/"FAILED_EVAL.json").is_file())
        record=read_json(directory/"CELL_SUMMARY.json")
        if record:
            rows.append(dict(task,**{k:v for k,v in record.items() if k!="task"}))
    baseline_rows=[]
    for marker in sorted((root/"rules").glob("K*/*.json")):
        record=read_json(marker,{})
        if "summary" in record:
            baseline_rows.append(record["summary"])
    for row in rows:
        reference=next((r for r in rows if r["k"]==row["k"] and r["seed"]==row["seed"] and r["method"]=="UAGMC_SOURCE_L"),None)
        row["UAGMC_LWATT"]=reference["LWATT"] if reference else None
        row["delta_UAGMC"]=row["LWATT"]-reference["LWATT"] if reference and row["LWATT"] is not None and reference["LWATT"] is not None else None
        candidates=[r for r in baseline_rows if r["k"]==row["k"] and r["full"] and r["ATT"] is not None]
        best=min(candidates,key=lambda r:r["ATT"]) if candidates else None
        row["best_full_rule"]=best["rule"] if best else None
        row["best_full_rule_ATT"]=best["ATT"] if best else None
        row["delta_rule"]=row["LWATT"]-best["ATT"] if best and row["LWATT"] is not None else None
    save_csv(root/"RULE_RESULTS.csv",baseline_rows)
    save_csv(root/"MATRIX_RESULTS.csv",rows)
    groups=[]
    for k in (2,3,4):
        for method in METHODS:
            selected=[r for r in rows if r["k"]==k and r["method"]==method]
            full=[r for r in selected if r["LWATT"] is not None]
            values=[r["LWATT"] for r in full]
            groups.append(dict(k=k,method=method,n_expected=4,n_evaluated=len(selected),n_full=len(full),
                LWATT_mean=mean(values),LWATT_sd=sd(values),
                LWATT_cv=sd(values)/mean(values) if len(values)>1 and mean(values) else None,
                worst=max(values) if values else None,late_completion=mean([r["late_completion"] for r in selected]),
                mean_LateSD=mean([r["LateSD"] for r in full]),
                completed_only_late_mean=mean([r["late_completed_only_ATT"] for r in selected]),
                completed_only_late_seed_sd=sd([r["late_completed_only_ATT"] for r in selected]),
                completed_only_mean_within_run_sd=mean([r["late_completed_only_SD"] for r in selected]),
                completed_only_final_mean=mean([r["Final_completed_only"] for r in selected]),
                complete_subset_warning="只有n_full=4才能把该列当四seed全体ATT；删失不可忽略"))
    save_csv(root/"METHOD_SEED_SUMMARY.csv",groups)
    progress=dict(total=60,trained=trained,evaluated=evaluated,failed_cells=failed,time=now())
    save_json(root/"progress.json",progress)
    return progress


def archive_data(root):
    root=Path(root).resolve()
    destination=root.parent.parent/"experiment_archives"/(root.name+"_DATAONLY.tar.gz")
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".tmp")
    with tarfile.open(temporary,"w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(x in path.parts for x in ("_author_source","resume","__pycache__","worker_0","worker_1","worker_2","worker_3","worker_4")):
                continue
            if path.suffix.lower() in (".json",".csv",".txt",".log",".md"):
                tar.add(path,arcname=str(Path(root.name)/path.relative_to(root)),recursive=False)
        tar.add(SCRIPT,arcname=str(Path(root.name)/SCRIPT.name),recursive=False)
    os.replace(temporary,destination)
    print("[数据包，不含模型]",destination,flush=True)
    return destination


def launch_worker(phase,task,root,config):
    root=Path(root)
    directory=root/"cells"/task["id"] if phase!="rules" else root/"rules"/("K"+str(task["k"]))
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/{"train":"TRAIN_END.json","eval":"EVAL_END.json","rules":"RULES_END.json"}[phase]
    failure=directory/("FAILED_"+phase.upper()+".json")
    if config["resume"] and marker.exists():
        return "SKIPPED_COMPLETE"
    if failure.exists() and not config["retry_failed"]:
        return "SKIPPED_FAILED"
    save_json(directory/"TASK.json",dict(task=task,config=config))
    save_json(root/"RUNNING.json",dict(phase=phase,task=task,time=now(),directory=str(directory)))
    command=[sys.executable,"-u",str(SCRIPT),"--worker",phase,"--task",str(directory/"TASK.json"),"--root",str(root)]
    print(f"[{phase.upper()}] {task.get('id','K'+str(task['k']))} {now()}",flush=True)
    started=time.time()
    with (directory/(phase+".log")).open("a",encoding="utf-8",buffering=1) as output:
        options=dict(stdout=output,stderr=subprocess.STDOUT,cwd=str(directory))
        if os.name=="posix":
            options["start_new_session"]=True
        process=subprocess.Popen(command,**options)
        save_json(directory/"LIVE.json",dict(pid=process.pid,phase=phase,time=now(),start_epoch=started))
        try:
            code=process.wait()
        except KeyboardInterrupt:
            # 只终止本调度器刚创建且PID已记录的进程组，不影响任何其他训练。
            if os.name=="posix":
                os.killpg(process.pid,signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    if code or not marker.exists():
        save_json(failure,dict(exit_code=code,time=now(),wall_seconds=time.time()-started,
            note="保留失败，后续cell继续；--retry-failed才重试"))
        print(f"[FAILED] {task.get('id',task['k'])} phase={phase} exit={code}；继续下一格",flush=True)
        return "FAILED"
    if failure.exists():
        failure.rename(failure.with_name(failure.stem+"_RESOLVED_"+str(time.time_ns())+".json"))
    return "DONE"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=None)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--allow-code-change",action="store_true")
    parser.add_argument("--phase",choices=("all","train","eval","rules","pack"),default="all")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--eval-device",default="cpu")
    parser.add_argument("--vec-env",choices=("subproc","dummy"),default="subproc")
    parser.add_argument("--ks",nargs="+",type=int,choices=(2,3,4),default=[2,3,4])
    parser.add_argument("--seeds",nargs="+",type=int,choices=TRAIN_SEEDS,default=list(TRAIN_SEEDS))
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--worker",choices=("train","eval","rules"),default=None)
    parser.add_argument("--task",default=None)
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    if args.worker:
        record=read_json(args.task)
        try:
            dict(train=train_worker,eval=eval_worker,rules=rules_worker)[args.worker](record["task"],Path(args.root).resolve(),record["config"])
        except KeyboardInterrupt:
            raise SystemExit(130)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        return
    root=Path(args.root).resolve() if args.root else SCRIPT.parent/"serial_runs"/"uam_source_K234_5methods_4seeds_60m_v1"
    tasks=[dict(id=f"K{k}__{method}__s{seed}",k=k,method=method,seed=seed)
           for k in (2,3,4) for seed in TRAIN_SEEDS for method in METHODS]
    selected=[t for t in tasks if t["k"] in args.ks and t["seed"] in args.seeds and t["method"] in args.methods]
    if args.list:
        print(json.dumps(dict(version=VERSION,cells=selected,nominal_total=60000000,actual_total=60211200),ensure_ascii=False,indent=2))
        return
    if args.phase=="pack":
        if not (root/"SUITE_PLAN.json").exists():
            raise FileNotFoundError("指定目录没有本实验的SUITE_PLAN")
        archive_data(root)
        return
    if str(args.device).startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("请在运行命令显式指定CUDA_VISIBLE_DEVICES，避免共享服务器误占所有GPU")
    existed=root.exists() and any(root.iterdir())
    if existed and not args.resume:
        raise FileExistsError("目录已有证据，请使用--resume或新--root；不会覆盖旧结果")
    root.mkdir(parents=True,exist_ok=True)
    native=materialize_source(root)
    snapshot=source_snapshot()
    config=dict(version=VERSION,device=args.device,eval_device=args.eval_device,vec_env=args.vec_env,
        n_envs=5,requested_steps=1000000,actual_steps=1003520,checkpoint_every=50000,
        train_horizon=500,eval_horizon=600,eval_seeds=[123,124],
        trace=str(native/"train_data/passengers_300.csv"),resume=args.resume,retry_failed=args.retry_failed)
    contract={k:v for k,v in config.items() if k not in ("resume","retry_failed","trace")}
    signature=hashlib.sha256(json.dumps(dict(config=contract,source=SOURCE_PAYLOAD_SHA256),sort_keys=True).encode()).hexdigest()
    previous=read_json(root/"SUITE_PLAN.json")
    if previous and previous["signature"]!=signature:
        raise RuntimeError("本次配置与旧目录不一致；不将不同实验混入同一矩阵")
    if previous and previous["script_sha256"]!=digest(SCRIPT):
        if not args.allow_code_change:
            raise RuntimeError("脚本版本已变化；如确为修复后继续，请显式使用--allow-code-change并保留变更记录")
        save_json(root/"code_changes"/(str(time.time_ns())+".json"),
                  dict(old_sha256=previous["script_sha256"],new_sha256=digest(SCRIPT),time=now()))
    plan=dict(version=VERSION,signature=signature,tasks=tasks,config=config,source_commit=snapshot["commit"],
        source_repository=snapshot["repository"],source_hashes=snapshot["sha256"],script_sha256=digest(SCRIPT),
        source_ppo=SOURCE_PPO,nominal_budget=60000000,actual_budget=60211200,
        new_stations=NEW_COORDS,copy_station="新增站容量/充电率复制V1，旧站不变",
        untouched_project_env=True,source_env_bugs_preserved=True,
        compatibility_changes=["新增站配置与空包标记","正确注册源LSTM并将平铺历史还原为6帧"],
        execution_difference="默认5子进程替代源DummyVecEnv；--vec-env dummy可选择源单进程执行形式",
        evaluation_scope="作者训练文件回放；不声称独立泛化；原结束/满座尾部不修复",
        pairing_note="训练seed匹配；作者环境使用全局随机流，不声称各策略外生事件逐位相同",
        physical_note="忠实于锁定作者源码，不冒称所有代码参数等于论文表格",created=now())
    if not previous:
        save_json(root/"SUITE_PLAN.json",plan)
        save_json(root/"AUTHOR_SOURCE_MANIFEST.json",dict(repository=snapshot["repository"],commit=snapshot["commit"],files=snapshot["sha256"]))
    save_json(root/"SESSION_START.json",dict(time=now(),pid=os.getpid(),phase=args.phase,selected=len(selected)))
    collect_summary(root,tasks)
    print(f"矩阵60run；名义60,000,000，完整rollout实际60,211,200；作者commit={snapshot['commit']}",flush=True)
    try:
        if args.phase in ("all","train"):
            for task in selected:
                launch_worker("train",task,root,config)
                collect_summary(root,tasks)
        if args.phase in ("all","eval","rules"):
            for k in args.ks:
                launch_worker("rules",dict(k=k,id="RULES_K"+str(k)),root,config)
        if args.phase in ("all","eval"):
            for task in selected:
                if (root/"cells"/task["id"]/"TRAIN_END.json").exists():
                    launch_worker("eval",task,root,config)
                    collect_summary(root,tasks)
        progress=collect_summary(root,tasks)
        save_json(root/"SESSION_END.json",dict(phase=args.phase,progress=progress,time=now(),status="FINISHED_WITH_FAILURES" if progress["failed_cells"] else "FINISHED"))
        archive_data(root)
        print("[结束]",progress,flush=True)
    except KeyboardInterrupt:
        collect_summary(root,tasks)
        save_json(root/"SESSION_INTERRUPTED.json",dict(time=now(),note="使用同一命令--resume继续"))
        raise SystemExit(130)


if __name__=="__main__":
    main()
