#!/usr/bin/env python3
"""未来请求已知的有预算数学参照：MILP分配提案＋作者仿真可行性复核＋有界邻域搜索。

仅支持封存作者E0与严格E1、K2/K3/K4；不是适用于任意环境的全局最优求解器。
默认飞机仍为LQ，优化乘客分配；不改变作者满载、充电、站点或奖励。
MILP只是代理模型，代理gap绝非真实调度最优性gap。在线方法不能读取本工具未来数据。
"""
from __future__ import annotations
import argparse
import base64
import collections
import csv
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
import zlib

CANDIDATES = {2: (0, 1), 3: (0, 1, 3), 4: (0, 1, 3, 4)}
NEW_COORDS = {3: (0, 10), 4: (10, 0)}
_SOURCE_K = _SOURCE_PID = _STRICT_INSTALLED = None

SOURCE_PAYLOAD = "c-ri}dv_bhkvRHQCgjNykU<Cl35vuqa#oRKulDNolVs=YA$hPMFeDLy0D=c4Myq>h%X-;*Igw>cmLGB~TXC#M;@Fl*J-)&H5V;(X{L8<OaI2~x)7{fE0A(kez31nIBB!Ujr@Okks=BJW>f}JvZPZ(pcD=bYkQ<n9w;Qe8_U&_(_Izh1S*kB=KiVwL&Q?mpJFAWP;`Y~ezPf+cKw<#iEL7UiVtiyIy~8bL(zEH2Y-X&K&bkxXbapgboSm4NP0wUgshNpldIZ|cR;q4mAa`=$@SdH!_wPwAl%f5gLtW;a*P1iMnzOUg9DcP~@6^i9{`yR%T4^siyXy63xl${(EA`qG{O{k~w%e@Fxiz=mX*sXE&32_xZ?>HyZq+SeCFfwHU0JC7mRdNo^`>*MHoV(iC_=CB0q1D(L<MR$7b_*Vm2-AFyIr^89Cj-;s9SOu+*;e&=N6l_N^Q>B*=RKD#nL=}d2rDA-=F;cqH}N&8@tD`jE3((9RQ=-a$pXXTHCEwD|6UQ*)3ICFpA-Y;xX)Pwpm<okJp>Woc4UN?NnXpsW=Di7CKcJbG=-wI`x?rbWekar*g4fba3QNZG2TRzINTIv+26dqJ%|D!lF`h;PYiR`yzn1jk9m@*(aT&^R6<gV!7OOTR0I|;YQUxk#NdOwc<jhl%PS(Yp`DsOXSuTE6sWhN7iy?mYiSL0SMJ4=%m)Fl-(u{v{8h9YjbYXVS{NUoH@4AZH8i4yIjFR!z*Pf?PA+)vCkKwD}c$OJ6|bPU8huR6lnw%m<cY%%jL?VQv}%8@>2sP7>(PU8u-Oj&2e6?EX+BrW(mr19PQTjPH}FbRFG3oHfnQI1J3bExjhe+c4U7}EnnUaUH(EY{>^3Q|8kd{SKMN|1AxNvVQ1U6{kVF=N9r9|9_I)S7|^K!6<`RoklVK9oV_rdxh77=Ik-CjgjU0K%jUd0eBSQsM-T2Zr;8sx>MlTK00|$Fv*!drvm`#b2iCB@)N=M6Il3P)1$v-bZPBdkHr!g-t(BllL=}B~3G{#~zv(~?`OQOz4i3*0TW;AoRIgS_OWJnBufB%(bEsKAQCWabSSrz;Y}RKwtu{>$zB2>MDR7#9XK$_4BxHw94!g&T%`$A%7OX9P<7lyT49E$9t=Me2usP^GY(wa10amd+@4)xm;a1yi3?HxfRa*1K2JJ)`!i?L7=_3ABXfL^lh839WQc|w?m!CSX$nCMSS(>l3VP^plv4*~R2-{n*1M0J|y_>G<EEInYpMh=NfwAVWMLG#HV;HihR;rirv%jp?OUG!rummQhuY!dh!0xI$Rp<mL!7Rr`7;*+XH!uOd@AuD&S|RH{+pI6})o$Z9b7?qbz5}8E5$vC@=K;5klg4geg@H1>W|-G<Vm=zfZi6Kw$`E%tEu1eP5W&~qQ?Ipv2(kh!x$@2rkFNY}`O@ERya|6eN5%RaY2#<+Ds1B&ov*ZT#eWU!fxj0k2s;LNxuh-GB;mL~#`zG+mDyRhiQr>6-qdz0ki}oR)%tO#h<gRr7&<OBD-AX|#MllF7?uIL@9;SzLeUreZ&Mpu_HEmC0msCauyD;zEw^pkRP8_i^FJF)?Rn&C?6=dbCL2riVyXrlv)C@wXMVlCP=Nm0g#x?-j_H@Xi|u-~$6NZBy`tqs^eI(~MB*z8v_i5d;mlUtYI&+gHQP(HPk5z+4TgXGCHqE__0Gw0fq{}Yc%2mzs@odTzXhOrCDDS2+z}{Xxbd%YTvIjr#VX{GnmbOVoWr$c57<jiZ;!=;j`#N6?(!S!Z+{2>@~SwVoHjZquRSLB?%ubDRW5a!h$97w%?13d0L3JHe;OJesMlOpzg=w3x&2iaBi#7*t?tTIb%ZlTBuYz#W5?%n&TO>~VnUuRObiz~|4Omia^s@C<f?@du#t8SNdweRi7)u?D`k*%sz9G)cV&0B2#l*ROJrgxk1yl?(7Nw`xB32G+0cG?a(?~V-L;j=Ypc)KZ+-H^lMAdC5NL+%1*EqAcvHy~!T;&+y$hFE7v6<Gu06fMI)(1M^Jh1I=XX?dPqYig#bTw3tRSkljq_ivU%9uw`upxjpRnG<kfBN*|8u(6A9|mCj30S^X7lzce1ytbtyvHc=G%obOtuJIt#E8X`w$0PwOFgU<pNT+Ib$tkrC$U8DioW|%AyPFJX5b%;VmtX@*XH`vDvm)5KG-Cwf*kX$rRfFSOb>GDHK4oZ1Y_n1IcbS&Nl%eG(0C2oPi#w2=|06N6DcCf*Mxvn~H_b+y4`<+46P&u-yIRPG3!M(3O_InrzI@@Gl4i%GjGOaH<*~FC!i`=VV|rCiMOJtDSp~?mh78DHTSgVy#dtmE1<VAThp0-%U8R&H`-<+X%!Es&0*HC$;?&cV5EQ&dVxH1uWIi&lg*TR)<RuZH_hk^Fc(jgW@-xN)woEUTxrPQ$Eab5v7dI0udEl&O!smdTc@M3B?7sCmv{hFH~v;-muU_9%x@A70-tKG5;bjzN!KRW&tFyG!H}(2!tJRTQ&l`>)&=Beic^OsH@W&+=xUJ&YW4l_N{m8u7jeUv-arT`d=S-Z-463u^#BnC>o<A-OG13o`2XklH(V}D!bKokp@-(Sc-)<r>jy7ofPO*1kniAQAmbZVRccq_r3YGrmQ#RNs?4#wX@n<8H@GGzW;!wQnysddJX3dAn8rSSBzG$^-@OX@&kogsMwk>Fp3_FtAlo{(Zd!v#`+EnMFq?+h$e>(q7^Mdi2np@^3Hs-dF}q%^ZVZCmwtG1bMu40bnkukx8+;izuX2)?ylbU9{$<E=6_rM$WhR(pE+T(T;x<<-ttWRy{NY`w>e(}9lsL(BS_GuU<C4zS|LCtm6sMNhP_mG7}fzXQmYFh?bN_YnqH-R%2~h@=X?>Uqt7K!BBwcU$>3*G1A#t!06pPM4eX&KI(=Zqr4=+gw)Bf^$KW`Tyhg5-MbbphVFKA`rF{zOBOHzIt6v&)HvjPP#_fw?GMgezi9KzwFO?iE6i*aL$pMUKOv+H<)ZOS=AK1p}r<>=lY<&5acm8)qM?v{cACw7`fTRc2V>+_`NRDg(wR_<Yz_9?F8{d8p0QN3@10@<zgS<8$HrN1WSa&fa(xR>+TIv+^N3&?#fl06_3(EG1*8Tx2dXf7;s#G$iew4FfMP?Mc!X^x&v~lH&wMWbAUtR+8?VbOwd;M|uoA=bk4Yo^`P_Gq>JA-RaZh8-%Y@GY6?YnD_KiT-~Pu)jn*8jK?Ej1i%cHDq?7VOa%fYg;1D@3c>kL)tIRT*Hg53aIr=Yic4dj_58HA@rV-#E8%|Jmk^OB+|8ZY;kKg^U>av+RdM^6%P5x@F|G^wA4}jq1kfogLgQ^(+{Ggyd6z$wTk_!;NRp`iH%~b$bEdCS#id_SWSs^RxQ-6_79%D1OO6Q;iVQA%jjXuB5<FCt-dp<iBW{Id1kMW~vVF7rl;KIonuzNsxUWVc=lC@!+%W<vZS$KlFWgK|EzPh5+5?H07ir@(lpe59Ay!O(o2?G!I#Uo&t3Odra21?Y=;Bb9>3wmVFm!tR8PfJEbg8Um>NUx-MIq4fM?;8yUs2(J58Ot7vNrGo4Dc>^7O!mMce_#ag+(U{%fTuE5}(eZaL6&d%CWB2rgN|7sf7Eh}fksyE#^7(;VOG$Dz&aOBXQT}Sjra+UnQv~Rt>=kU?JhYlV-S|F;ir*L@ZQQ8cO8u*tCMjUBDv>p)RH5bH-wIZxjl9e5HTMabM2<sVdHjL(@<D{#nRxMH7X_Sj?vJ)`lCpUpOF(!8;9HG@nIF)i$Xk&-D6@u9vaA!GWvR&|vUaiu+Tkce$hCXmr+ove{6nh{$`AB6lE;nhEg-SUfx6z_N6GRWJnP7m!`X^7Bau+A|MzQvBc&|SL8Pp5hYRh#drv_3}19&1zPYq0~GeZ?oP{w2s*gSV>{po$S3hH9^8X=pc5V15kmYOcI1FFG+$?#O1J)PcqdeYRXW(R-7>(@T%p1<l{dawKOclL3!ufbl$flm$m`@vuSN8!k*S_7@9g!p-NvrPI8r~SGrivvt5wC5`=pWY$+u<e>jb&g7J{Ht<T8wZ|BIWJ&kP`QOVu;IKIc#<B?D#cm{O$wc++f-^QrA7n6CjH8yB+xTILaN>JBBLtM%Q-w2_%%g|QPIY${YJL8=AcZ~sWLe8mC-cv{OETRS<Qr{T@r$|jz*x6(`txJtUVfZz^SyHTD?vChmi!vyFf^*S4!;6p;MN4K=L5G>>DQ$NO}CngoS<ZwqnclNw<aU;qOL5{AWf?IP3Brf7kuyu6On#NcIlrbKy+^`eICIJyA)&xg*dE;`L&+<L+rT>&@7#P+CFDlcJAPj`#gvyn7#ju>Zr8H>L*S20ar8BAvX-h0y>Kw-W0~Q5hjbe+QSJm743&bQ{$o^MC4pO`vECOR#bK&f3#UY$1MlauFc2e(lfR=hrt+-|b$xy7uhD@D4O%mV!5-u<@ry;)aW|ky~oFw6i%ggjQ<pkYhSC7zP<Cvt+2YjVQLN3#R1Bz~n>?p^}S;Jgr#Vz$_*Nk}*Udqc%yD&wT?g>(RFiN_i$Qla1%sybnM2t~^-3cGr;z>z#jp<M!Q<H2@TY0}7u<*d*A_Not15__R4<e=TU3zA)wg1v2+^*s!}7mb<qu@L_W+zkSRnO_(MTdrt;+^~ymDD$(Cb5K_&vR>XM=n)Pwo3O{i{``+g*{mg}$O4d>-v9{1pUC<!v%oTR-f@Wc-E;@PjKn?ATrsIR*r=XG{JKd2yfl1$CcyjTacmJm1{kMMet+hw*tUX!=@lwTo6MIU)TPtNcoa90FHgKr#c}h+aq6ITpKKUe?d&lo>T4PTr4EVz|xe~*EUB&CyPPX8Sv19I1zFJ(EDHoj+Ip>6iL(l^Zt>xxXDHfC@{H`Wb(_sg0adK7sq_*O#GzeV7J9B1j<@v_>ublOJ7of<y@@DtFhw$&l@@Xf@{+p^PQa&ET#2HW?4RThAT<Q=N>}PT~G_B7pIoMQvrwZV_%Fe|KbD&l$3aJ*lAJy6mOHJO%q*BS0@=i$6_-a&tWpy$D_B1l0Rf`sSQ^{;r9ZEpBP9-yHZ{hN8n%9aaq}^j+-`;(DU)x{!_dQ2n1lId1)k<f<`FFP+1#VU=49;U=a2^kZ^GK>cIFF_JhclT%#!|f?o$U|O9MDs>Q}+Eurx@B>iUtP(p9|G{puEHmml}K%PRP+v2WUdVFX8lYgtG%6;cVsL6$PV!ZicWdV4kJ<qW&9yD@G+oy4hId^7UzZWF0FI#Qv2>(}3q=s&9KX`tp(1|6B8+U|LTffyJL822KNy`*2$cEEO{vgn2Nj@C)4zTY+bPhiv6dknw0fvkKpF(cercP~q={Q|z?szVcHw62!2Az%BAP$Wv{QZKXYO{f{e~@BamkG9W~3eD<e}ho`-jFIWriN2fO$ME>PHJG1uq1MmK4&UR;Qb=7<OHf!<AgF6rJHk&|;8>`)0D{GJLv0BU;XpBYp63ANI6lZ>&S@>ivVKy?p$P7}l*8ZJ)4;<ZdVCR8d3J73atAFjj{rUE_N0*oc<*&g)u@UMwjql)x6%P#!P1SaG+Vk~h&N&2lyz@V%YP-pWIFrhZ45u^0sc|Qh$&F@n>G7%BzGADrr(9_trDGdZ868fghcg-Ag1K}yH#*^0sW)3WrvY6Smw<L>Ml#uv)M(~kfAbr>b~RO_FLOkIWo|4v#g<cRSS<ki$RCCw$o^s@#txDKGeT*sm4Y@3t;%m<>~tW5b0?nCCxKIprKS@h4Jr#v$z87r_v6XQ>22v$D#3nD+rC7?F*4em>5$6JWOUXi?|bxmfWL~zT)bK_8>*ykG4FkptWvqjR02kv&P}FMi8TC`$xUWQ6Haz?`c$tD3b+S>??w&>{wcd<wF-d3LD&TdW}{Gbaa0nzB#H(qskVZ=0a(p{<iWMIXYY6qSDf@baPh2n=IhOK7aiDK>v!KlBW-%#J9`GUyfc;}UmM5pKY4Ja`_b)<XOA84vyZ*EKkdGA$NTi#^>0=|V1-uQYme6+t!#Y%VEyU4-iOa%v)#Y4@#ULLm7o$bplz*sEp#98*X-Apc2g>sP3KbC(A|{64VTU4GCQ`mn?iO@f?Lqw9M2rD*+x1<1K$gaVlgrul9$==t_<)?JZQbPbHCFnxwT@mQcwEzDvK_^WwKB#%~xt}vZw7Q)Qv9aH^)>>zM9B?Q*T!0fGN`T&I;~Lxup%iJbt&*Rx{{jr&yMqy=6Xo!@L}nbL{PpG>UZ5D6F|L1L(^rG3whU!`<S{;;<ndC52eXczjr+8dN?6M#1iUii#HZC3Yz)ssj3Y6sr+6xIa&L1rDm-zEmoBthMcs+RdjK3atefHoE9VbA_E6V3!`J2Auzb2iP|O#U#5)mNV3B+;$nVbj{)-$j>czj@?5I8&|tfMn{@Lqf$E7X^@W#5V>1-(V^(fB}E6b_V^v|>gA2+AEFQBhkxx}e;DZLcoFt%r?_A-O`=cilC_+D9>K2O)G?=XxKnFKA|rPjnHs3M#{rTWgrNvzH-=^)o^J;-1rUe&+Z~2DRM>PE>Wi*<3~6;{3Zipu=?0xeXha>u&tlmjorcwi3P;{FXD2nz)P2j>fGFz5x~^(W1)MAd#O-eZ<mQW0fguRlyC;h8tIxHf;a|WDl?bTG>7L#?zQ2y#I(SvQ2}d!v&!)&E*rIX~j_omS_^|I8#fA`%uz&0-R;z#%s7>N54Az+4^$c&M?{oHtbt9u#Ub0ebomN`NsZJ{`75t4L@ZIOCeJlpM1ISD;TWqxUXxTgS=kBcw-nmcL9(~31`wnfs?q6<0Ih?pSRgrhy>k5G}EkLHt$L@>Ea4uPF;MLt&3~MJ;pDSY*$K%$^9CDKj-LStgZI?Oxx08DNY3C)<07cvvQ5PmhH*WAq**$-qUFGw>|I7O76$z^K)f>VwDR|?9uJji;G3$BbV`_lc<lm(;ZTH?Ekl0_i<ULzn|M33$&9}T;58>a<<!kalsehE4=lcJuAhGIa*({l}ru?ksYdT*K+Aa)~7bJWu7>|Jnns}G|l_UGNKzIz`3rC*N-6&yiLk>YS@`^BE!#kB3O$`v|R0&Wh;|Z`@`=Y7CATgx(7<3l3)uAw<BX9@PyYiLy#!Y_HQV}%>?gst^0WW5wf4R~!&(|+DM8iyjN}3P;;yt_FUH(#B3hgzNUj#03St<=nO|t&gUg%?6w;-9z2iMMW5o*x++pRMW+okM!hGTJ&iNFr#q}>chkzQ*c<(ryqbpQL4YwtSUd!KvfSGyN3p=}?i+1sCbXYMF$W%Gkiyz}qdnq$kiKiN3_)FEeX!RUeRF)7#k{H^t?XKgKYO-V~pn&5SbNm~=&%N_U>8e+A>g{{SOvREz`fPMPc7-IY!88_jf5MF(aEh~`gYw3K+(gynm1tQQ+>x@)K-<naGLG&5VJYl}XF{0kaI&`+77qR#jNd0SV3ed9!Dj)n_h<A9*CMD=ctCy@*<eAC-8c$6xMfA;xx#F%Bypo6-I{X4mqtXahrl3A3@=l^kE-bJ8?T`rSNlZ~0W&X13fRydDwS%ft?lg;j2oY2w;jn`}1<okWt5T-q09X7$^%KjWv+?`}h^XBU9?0lP2u_M1h1f>AF{U!mWK^^qtX$i8aA)o5b96%Bje8xbxsO<}T9S9PZ1}=HXq@5#%MyB%jzQs9fKPHB#kK+j3YbBWY)&-X5>xKRgP{4_%}+7R3_u9L(*>G(>qBq(3Cw)$(IqNbUSB;AB~hn;g*2j@2>YO0W>sh#^*>1v2R{51b+FGrz_SI}I@a%9fB~^Z1i`U;{jtLs$2v$0m+!#R05kCJe~eBCI8qy+ULogo$JZ+8w{Mvp%$P{A(hOU{z!Z#iG<J`5&uW?yF$jKZ6gRMbVKj6H6lXDx6RPA95u%hnT<g*sKaJ5REZD0k&U&Dzcm5;q%Xhqs*HPr=3uhZ;Uvile^<utWZ@K<B|9+%S9T*)=dlS5a2lo93@QPBsS(ah&oPTlnmc(#%01HKbbl@Pv%<b~Jt%K{+sWOYmyxVk>QAAPKMM$k;m^vb;dRl7GIgwfx4j+8&=pN09gDaRcKw|w~+aRda3f&k%Rh3<d8$mpJX><eM?xB!T7d}!kqgPb7fIMYJpAolk<WhU5t<pj_U0-qK-TqULjb%}TTcuPhti}KQt0o#9VM-j%+En9%Zbn(Zzl^s>zW&lX_vz-ji{7*MH@^R9{fkwFD4)=c7n7-JXP9@anGnlMM#Z#JF+Cj=-Js`f+nfwus$8PK;`nbWnKj)VmDqD3H+TH!*IG<}czB<4youVWW>077z$Rp4mRIKtb5CGrsT5eWu~~aw=&qu~LBF0T3Do7PPf?LQ25NB?sHkOXL{C|)tL&G4C2(kH$mz$9n$C@+bE&bOUJ04e7xGFFR}%SSskQh**AuJtxlXeqYQxhUB94AzP#BTvm#FF+X!XkIyn(iOUTuG|5j5+DUmx@>yZj1am}xiqPN5oLgkK^A(SONdfiqzeL^h#kHgX4p<7&4Ll@KNT_Mv{s&}FgQ>a*&XY9KAqwM9<FU)<0P0L<ZX0bxOpl6b&v5zndU-Gpxc!tG-;V3)>#oNDB>Y@h~C+<tQgiR7*$uM=osl5Oq<W^)`7J`~Jv!s@Z6j#s*_sfAWsFOFZ@dhx+pi$<7NCS<jGS!{s}p|wkzK2}?AYGUsNjEu<J(9IAs*W=@K&_Y_7RmP>>&9W)v3b0V+f?q%8|G5>d!#?OmMPJ$w=-$Gx9Trv8r1d2qmmKYg1gH2X#2-ru3TktL{o0BmBRCi0%LG8gCg!5AK!JkB)<F!2$vO#Vs5z8ifly&z<%c@$+2Qe_IQnAE20Yn|Ws)s`5GHXk4pTvF)^M|Dv2Tx~-Gw!=Xs2+qz{<v;-sBL%WoSActOZNkrsWhS93MF>4p#vTnq<CR(;><+wVp6GowWBm#1?BP=S<2`i-m-dCmgeM2}-T*WZ0R)z_lJ3BIx}wtiaxaot@eW${xnc$!Z+yEnu4cVsjt+-n_TmkKxf^E?XETZH0h|G5YeA|7?C3H7)23HwQ;k#?6$|*<ljzT0V<nQVa;x6!sk1l;ly#9_D4MB@A`M>`uCGc;_na>*v;1zhk%gx_57TPp+tM1JsZ+(=sYomW9HH$Gf`Ny>(&z=3DF${bx@<WoBJw(Sr(`x0W})c*7~zI~c#ZH1C#<S?r+x2Y;5gkzU-#ILY8PwrI(bd{h#QnnDBg7pCCLF>}*GkJIrGLeOAGo2h}5SnX8dWaZS<04fRZe~x;m`_J$)l2u7ht(h2R+yrH%u?BkXvu&QoL5e*;G_9S5_&as4+en<bV<sEy-Y9#t*XOt)XjVhJh-w@_cwN)9&~kIVn`nX3ZeHo$TCrmttWW_?u#5|8v;fz#K#TxA<YOVyN<4@z{6;nV6^lDBn=*@K_}VZ`n>}HbKN7XB`FShu7*%>8NZcZN^s&I!g$<l3jrHk+rtYHg!R*Twpaq>`)hBaXkvH^dhDz3w7MeOXwol9$ALxZS9lVpIY@DbTipU)BXXUuWmHu{XF@G7jTF9>`2_!^*j6bPlTEu1Ir!aJV<_tTP%-CMW6Mw(1^`7->i3$WmCa>1$9|Gf1!<_h05O{&E0}wC~5oYFNM$ER4Z2=dAl`=0`bi#%>QUQ~<T6%1%ntX4qI@bmm4v+ytAV^66#!`0fMibo&30VDH;}4a$BCg{LhfP!1L849UR_H})x^!7-y9>eF`juH7fcyXvQ+$aMHo1W};+|?a#DGoS?3(QHWvl&NOsWSlm&Wt{SMh&!+~!hmrAS{ZiYn0K%)D3-&3gnb5FrQ}JOBY(ps=+ekg3ojRNjg&>8o&T;o5Bxr3yd$!blt$^Z@-yEpg%+$jll^Q(+wM0|1)NQ)<{q%_&X;Jr$kl9l2CGH)6YfmBlLQvA*@2zN!qbkA1Rn|F)vXpgZ}OysH;CPouNAJQ_IOrEl1+Iq&nEYmf2jjd$ze=0~5&NA{M-d!<6AlC%YP+iGeJ*$b*pDX`d7jsKaPDYh!5UG>^*WsZ|}%s5rei}!mEymFA!WTu75PYp~S*|q1u&ck~T77pz^a%9heSN9yAc77gPa9f!3B;Jxnv3VGE6;?B^a7NMbkv0fJ3;)JyXxc+j;}tlmi1$+iV&#3upCnYoIAVAi7DjbkhN6MZ5$D-s9A+nnBis6)-dTJ4dvvP2dU<{I9vsTNx8c>>JaZDdchdF4%|CtMt$cwC^t+2}ExXqrdspA}D=dFP7bKY;8l%DT@d9%1qp}z?5DF;KxRh0xTUw|A)H@5PBZ7ym>ck^e-@*={tH5Q9AU(()A&JBwqb0|okfy=-e1Yctg|!1UYJ$;F4c_--;}b>(9ApTx@%#qfjZ;0{*rhkT9_M}WX7}S;xbELRTw6K2_Go$I*;$S;&gPqcSbKclJ9|kz+OaC^j;gcv{G*KrZ@@bbUIXUcf4q6_3O@<Lcm2K)xkypKT{$G(P)5jV0w#MOuE62j`}iI2%J<I3)2Da~H{-0WtT>K}p%|FKIkvEU-f<A$&wu4)@tNO@<$8%s7OzE9*y0Bp4^MBr{U$0%K6uc*ey)4_4;v3wy(`yg<Uo-wI8y(?TuRUl_1a^mPQih)XWx~9O~YHh0Wk6&JXYLpm@grp5{r$&J4YDYx*NbA-vN=;#?LVw3mfC+^39Fs=eTNz6(ukm4Eo>SzsXZKy#F1pDQkvX`|KsQX6VPm!XVvRChCBRV}vzGn2ckvQ<HmYi*cJp2(xLmDht9BK~WM)+t5o&F1uQb_Kq;u?7^H+-0F~~_3--=bq?>7(n*VwPJvWZt&pLpx`h5%c%{H(=flpCPJ=vkI0GSb5|CA%oy})lb1XsEk2DCK@V!6K){{+H1N8gfDu7(4H5o~zg6H5~R-lBgNMR9dNH-!8PV}GAqF+#a9!ADx?j4K~WAW8%bsET}k`fB>pkoQ2J~tg%+#w=CfEta+_X6R$3hDe;-Fu&j>_#CPWmW&gH&V;4XO?F?UM;4XtpW=Xx8Ij)`7w$`Kk=}u)U2+)%VS~d2X(_<1wzOXiHEJry)#m(2NOE3vQAL-)LDqx^@B_ds>K$WeCkI>Qe5BoSc*y<q>!jdfqU(dBSyuWI#8o>0mSw~WZ+h^0Z2cWGk3i|FN^46_ivN$cIfSP9-E?3KTd*;HMCd?of9&t1b5)}p8tu+aj;?<D~`yDn0Dvh1&dGj)If%6W_i_PzY>)V<?<=rXHf}PCE}C{G}7B?ktFj7uUI8+>NCT=*D->|bcmioah|>Z(0lq`_raC5M_=)Mdh;!I;hw$HED)bQ@t$1CIVV+B-z30$0A*P`4G+9}_~2^?cK;W-s3n*?O0zAZm$g#~@{B6Wf^>p=p25RnrA&|QI(x{RO5i<AXWBVs-TcCpn|b1CXeG_YN*p`ifUHhwpXqAg?A_gR_{o@nt#B}77=iP1&ObL>>^H9nZHLM3j2lvJnMFhn?~8|hQTPCbc~PDNVakF{4AliIOt#=KfCs6VI34Udy!YtdT|4*jBoSYAFZ|xS^l<Iboy~KXHm}{-c<=zv9DlyDaqTIYZnl@*na<o+j`>Rq&dx)7U&tznHv-*qJ`g3Eb3IWJs^CBIEjlmo_VIvTDsbc%zmK^~cC{3=QO((Q1P(D*TPq7|f(0=IQwt8?Bo_7xJd~b%8BWQ4D{n7W)@$d4^rUVG)ugQXxZk_Vb5H&7<Yqgc!t9RTne!lTcUNz-GYmVxnDf9Wif#B|laW!G^eOXgQ6Uxb0PGBs#|Sp$@GE#bODf$)9ivU#M_=?6P;n8z#+q@BdCvu~I^c#}b2aH3FTtAvxbFAGwP+}YlMge~c$ULFkb}W%RA|gj4e(eONB10ZPRjjs3JP`}KFYno2Mn+AbduZKyVoBwextsp3sO?Do1A7mSYFDLCp6o|IgwPYx5%0c8sGenYWv!EEJ$*-aA0=^QQ1AwM!m1nZGiVGbVR9UApi!zCyyN3`Ktrd*fx*4$y20I&nv)}N}DZx>P0`JtO0jF4|l_o8dIo=qhSp~RQZG~y0O)uwL}2zG7VliJ@dxK{YQ8r53CPqO3r?>e($=~@5m(8K_+jPl1ud()a$qwk<w1q@gTe;yNwCZ#BDR;sxsDu0WMZVMVpG6)<_R_E-LpT3#QCs`L2Jny1x2}%!-Q|lWUKCVMWX1)z=qvZWL-mGPjn=rFt}U2G=h?gq3C&z+{9Ip?6`S((Ej*=a7^MvkRm`^+WpZAZuibW6n@ms!L2@9%Nd5txqQDXkTTQch^?lSzG-M7O4B_Bk%d?{+FeH^IEi-E!@RH+nY%x@`Jg=0RbtZo4zi=h8gv?N<0?*#8}qBSZ=L^Y;OgDgI9l}ypNd7)Ubw~HjjmPtIb-`dfL-u8EfA_`lV^4u=DWYy|3>n$m|=_j_t-b;nUQB&no2j@+TMVtRrN&!Q^!$Zb_!B?=F7FSkqdVIPM3iH&SQnSsiM|2Xmp!TF=(`Qz6};YKaR3O0Za}qpx5al@&#uTm<_l`0ZVOvn^~4`~ad)(snutZ8|xeo}N_o0Mot37Ca1u7}xtpbrf?5!*}je1)T^KpIKt4mse6sL{wKbY_Js+jEij6_jcJLGZ{&8E+PZr{)^1@^dfvc`uOw$1I&Un^8Ov~`TNM)?tKoYIJOrsyv*Hiy!$t|E`a<vKGzST<C%_$Zj+{yC!T-ls3>us{2O_vls-nk&(Irh5ohITZrTZz&aL=0%kPFk8-<F5*A;BnuP;il4F=BW%<$-s%y-y(_C81`VPs>L>G|>}5iSI_A$%T+hmN3KS1Y&c5miacsMhw?juEYZfT(Q*J%Scyg8+>YVcs}s{1{SOTTqSP!m;`2k?F9(Mw6&n{VHjJV+2LD8dBd3O^0?M?)>YGMQKFCSvZtvZJ;2cI#~&zZf7Az^`-eC1nyg=Bx6bEuzS4NEN`)8lNy=nc7=XHQ_h+<>W#=EW*;`)VMKol`RKu8k*ruQ*#yS(Qr?MSj|STvc6gzxrk@UAdvS5jm|gG-+sX12m|5JUsRZKTyxFI|&~3N5fS!`%8u$uGd74zvZgy%V{?b?dh&-jf5i?AWM;Z=PTJ1Todpbxw*%<RD*>6+T;eXKdG!IerC3uGkWmXA)4hhXD(a(LJeSVb1y<3Q~sKi;+H4jYbXgF_6n}->6F-jvTgCSUg5O52T2wy<qQJJx*uUG_n0APFc0U5l3TiwEj*+WVffS5KkTw->UVlva=QMv}5$dG?*emH`5Gp6)w&fAnJaK^z_Ivj%A_eJJwJ#EOC29C92Fhqq85~eOrC%Ms4QA{&^oEbPo={h1FNN6CCJS$1q=%Zc{pDoqY&<gqqB~msJ6U`)jd-lft+w1q=U%&9j^{Z#Oi<D)uZ6nN()SaQ@_yP}*DWg{f2h1-gahOGi7%;rl^F6|#F<4G3#lc9`Jj4%1VC&xcn#ICgdvp)4o1K4mZS`yK@ptGcwS0Qx#=9HeKk&|e>plIm_xnre$HZbV+2-l2J-xN|Xhn}NDm~5Q*F%su5WlHyjRGjp%m}6-7esioQP89doA2M>{Oki515S<P-R}3_uRZ>A7$!(GSYP>4wXMTR2WGvta+y2iZs)85PC6Fsh&Y9`+%^#QVy6m+5qYx#k0cS-TLUE7nt_bl=AF>J#O?J?$R0900GzN}6|An}8ejx6(XYLl00#K2JzD;Mm*2pkf#+{{7q82sr1xZH{q%X%b6>h@hwBzZ^xb=Q?LD&h;DH`ex>$lx5tv+m(Yr28qNbA)gL&ln2%#Qcp`SA$LvSv?3w+Nd)3ecsAOZQc)yv#LoJcqe(1>m@U#@Q6_}nCN74^bwzZRAdGaI|p;p2OJOH$N9-<#i;qW<2u2?7F4k4*uq!<e+6rkyiyRlYRa2(BR;Nj4JQN|1h~ofa@HtR?sN60NvxC8|YWF=b<mV?e#Jq;(NUT*xt?KGU*_vVY&D39L*|tP0*9`ELULsi4<Es20z>^H2VSYA7&;8C$#zb?G5Q!8VAV&fbMfER$3B{B`a{PHI2kt~@WKHS@7_ptD?F3}%N^eOAJq9s5rc8Enh*M*2xh@!KGqWAZBI2`3Uh(Iu#m%bPSqKT%mYRQ1&&{gu8tX=8rIRc-l^VLG<!P>qMg=Y1>{v^SYYjR=ywC2lG&;p}0)pJa2%%baKwY9N%OnP>GaY1p9ynUMXY`0JEFsr*R`Bu>#CI|jD=Nfn5_b%e$p_^6xb3~PC=Yr2At_QI1vZ>m$FL6;40+`O&yB_|!`jli9xd7#Pu5<DwpB+T$_BO`hOX4}@xS*~BQ!6bj(fvK<$xU?-vI~1q=7Zd54xv^NiO_pT2+iZZhVLKZm&>(JuxbxIIeTAvHyVo9jS3a;c!eAM0RRqfjSz&Vq<hgIFt>_C`!(PV6ehrYec0B%ewDnQd320FaDL2VfNYIMjrnrEZP1*PaeF(Fa1@z*oaW2`f<yp}(M<cbgH}?t_xT{{Rx+Q+a%i+x!m1`+ZcDoNc2M!($i-=~r<ngdG#;O};Yw64!>&x&R51&K803W=b{Q|p~d1jGEi+BDe{hxA10H?R;$hhM&u(o2m-DFcjK1%zC4-h#kBLJ5D;Cp0QlDacOD{v)vd0)H3!K&E=14E0T0vm5w!g|Ckn>Vh+X7-++*}T09$0!a_)SQ4+bj9#a!{&HzFxXkbiP%=SL)aw>WnXaMv9Mr!_8&U>9}52An141k;6(W>_D7HYuC(YyJRxlP49Y>mr6a%4pagpj-(@-i#e2vJJtZb~`;$8>09o8~;Ap4<qAw4HSv46wP)~LAec2cDC-o_w>Q`XvDKrVaLif=b-|@}&Nql;C?a`%m_3pY|r(?m;po_V|lf|}>7N=l6fjydJLHTLvdd&IjGOa6KjlcG1c4vEYd3EF2yWUxJVz3yDbbJeh;I#>6?3^nWnU1-RPTT`A0PRuFigf>yQ3xUjMF&$_S7s|EXYX#3s@Pp*yypzWxLTjVV4o4r;)9OR<tjIuNMC#Tj(6n`sOx=#Sv|TJ&tWL8y}K0+t_Ex=!ZNPHz|eFlTj_!n(d8!0+lAG92*-0&ZZ9l0;{Gq8n=u@@_R3RIp6QIbBGWU=`^;Nls&H#&=xrh<PN!DEU<DWvDB+}%BXNINc=h?*6(^JW`EWM%b0aX^H+Q|W7lD~-x8}U_?{D0`8-96XQZR&RF+{yrj399D4)p3@cMSK@wcIM4R3R_pEK<er8y2;`4vXv*Ytj<i`Os2FB8<WjE!FxyS^9o8DrTSxz*?7MfT*otN-mk1wKnk&tqLZ|^Q+n*u3|MyigWI1_q~T3U%bI@nu`<}-u;i)FJn?%<~+@!dTc!XV&mCa@9}rSq31jteDFTt^ABOiiafch?u{8&WHOb6GviXEFFzrx1FNn*QwDLmk>Q8WO(#>NrlaTRRDKjT-*xsEYx4lQVoi80ia3MpB1|iSLdhWfd8atjiqVK!0CE^A{1Jz9vi~A2UZhoHtnLjVd))5Cc(^*2+`*q~JV5r9F8nw#e0f4s4~zFA2F6yB52thKkz6J+Xm%zyvLl!MDU%Ne^1EBY5WFtZwiMe%WLupQDi28D^zSqA1(ksT@aEI-@k&c%Am#@t<XFBi%L4f$xA^gyiD8|GQbT=ZA;tq!3Mlxr784p!LdEzjyMmE*=9-oAc8s17`@cuC6!{|FBObI&KCG2)HK%ZRMXq!YM1*#k%MJbxxWuTkmVlQf7XdAnct15zfde*XcpY@`uLNc8!C%3KO&3+wC<3v-3o`OHHkL0L8EDrr0C8#F(IT$(Z5v{iYv#bgq?RowF*VYB=_ds4Ve{goGJ-<xq|bvG=WhalMti=}5~ROZR9bbUT5yK|!bM~+v@DsB=M1(%P@t(NbaL7Ea&<3W06~S_WZHU6JH>_<l>vfL7r{N*%p0VxaT`vI?q9|uAZtmb$zf(Qb{c>SrVkX|85Lj72eJ{`77I65m>StweKm)0A4}nVf5$uXefRp~?l<pk(U$suB#ECwk}z$fp%&^g#?<d=2j$!*XAh|%`HHlG>ii?j9i&Ho65hE=cjO0@LTf91L&$bUu^FDjYFg$&Z`tL*uvCFqtj75><g-9_lRlH1jO5apFgY!~!^w=GoR-f16v<5%DE&n5NN)Y)G!dQ#MkuzeBr(y$@I(lUJco&yvZP9Kqn@z@CxvPb=@_AABf*y%SvRtg;Cqoy8q++0P%4}gh8K#*<n5WCI28#s6<Ad3EQm-@=~N)kNRu##zfMS{z=JEu6|X<^KKq!R>);%yM?s+0l&F2NSd~49)|)r4dFMVwDd=~f@Yqe)AF`faFkOi`CrSmO7g>{#G#TgeAFQ>+L}X1-&~MZafgY;IGsm`I@)1Q{q{2kHN-#ya5~IhQscEqgIB0|x=fyE}`Grp7wh#nj;FF@#G-W&Fd#`Wi4U6o(`_y~)ZdaW0|4um~q(4$XBl|xXbiw`c4@h0XwyA;d{%BBjV<6|o^K0IRALBc!#9pHXZtw1Z#_-1{tq}Pies0~UKMRVh;h6|*aj$%e5I_lbj~f<M-bxWmjY3%^z9kG=v<LQ=$g{|lsDTW!3TTA>bFr_`{%X{HiN=H@VjHN=ZMtT~qM8W0Nf|-dQzF#*NIy~Bx7OHiJFO-d765-;z*tT!5fWluKF5=~Y!(mntDJUx4a4T@2OIZ4XUhGxM=Q+hEU*$Uu=xXq%AYQ#di35$22Y>m+nAflZu*smo%JtQy|?e8N3I|S-jxSyt9Ld&`&0MRbLfYD{?pAHpQHEP{fig~mXse_4+JRwiPbOHSMPcEKJae65TQ`7hFRST7MTHui;*>ELf5$lcP0$s0!i&$9}k?H4Gi%Q#5rQ>3JmDp!u5^iGwWZhdQX<uFF)`;I~^YHJ-Ew**Db0R!VwW0ev8>Yx}V(Hc<|YlHCDzs(Q+tUL~66BdX*|`rMQ3U3lwnF#jPmFS)Z}y7kRAt9-pFxRhN|Q`f1&i+T#|DA)Je(B{V!zCrpQBDX~|fqkG}lLSx=nPmIcIp)_^a;t@N14<D}*fmiA$X59917t>Q6uRDS+qN+mC1UBbd)|@9fXYxeCS(*;bSYfZAP6x)(gwvN8m^?q3nqj-!lB)8E5tS3V7AN+Ld^*A}iSowA!V|~F9m@4eJNec_@55)^2iMk~z2iMxanken#yj)%=DCYTe%V-ff?3|7VQ>BVOYg}Qmecl!C+CkXXpsqo3Wp^f$K!Rf^_ym!GSlJ&SY}%I$GA@+a&w($M=q^T1!9E-%37m-WxKLA){NpC=f3eCz6(b?U>KO&>y78$nM<lz&bQk_@kA`mxH*lx<W{ON<J<9gTvc~)%V>nbO4RVf&1yLicDt=ov(jLA_qyU!&cgPi^uqj<$f(Uu{Va=JUly;?TqYfzCU$g3E}PkM8iiik6W;=+hrsVAE<AJ57y1wm5hcBmx+@s?W?ZL8(Yk%FH3ew1Ro4{;u)WZ42k>1I4aGF!{HEh}+(ZD>39_oqRx5M!ZK6J{WN;w;+tQexAr-WZiBzj(7|XCA@QP?xuXdD(+X+?ZGw&q!u*C`6x!)Ona*rM!1?cr+(D6ZpOeqBHd1QzZui2EyLcmCyB{X-qB`-K3C+|evyfj)F>7B9w((LGTlsON_&KH;yalLDTJr08Xp@uAK1!gIm)#}F!vZ+Cn%xuJdW=ieOS3!Ca+Jm&x{gPm;TwM~SU<IKaRo&vE8~)lJmOMy~f&u2#vB^N!fzjz#Ih4B@W}<fu^RS*?0E7CuwiLZYK=D3&C3?5T&}QP!7KflS41!G{<e6e14Man;-rCBAFSiW-q+I<AMVyy3<5>_N)F=*WVkF-b6)LGo`Ame1CR9OjRirIaHc8C4D8-Xu=0UMpLKP!Ho{h{pFzM}Tck%vF!H(6v+KE*S_$KBUKMS5>cb3b*Q~2k|KJEJS%?2H*#ql?m%nhk;Ae%kaZ)A5v6YE}FELN%%BHNz|t-eg(z_xK<K8Lv>4yI9aPqZD`5uYNU1m*oEpLX~y-v{POuJ!VqqmYrnLDo%HTK<Tw5mY&fxB;Y5&AMXpIB<T~fqe0TlizLF47ItFb)-J)!+<X?X2mU`akY5Nh4-_88Eej-wpEC0n83v)Yg$8XcD-(RGP9!a0SsKMWH|-=Dm=#kWtj?8mPSp*1u$5N7^`1N=d;>rnd>mvYV@fYQgZwe=y@Jva{LI}S;31Op%}*ottnTuAgR5H%Wh%Y{uk#JCTbj&x}knZObLj#IQ9shk=7gn8X4u$>INVWI^t9oYh>6NvXN%z$8Amqj%Zk%=EZ4V93U4W?p2>$LJ^=@ws?JUkPcSbE%hLtjBq$$GJy#ai|Ck$kPhQfM?l3F!50|gTUc<O#;xD=4c9|SeyoxSpGm6*B)%kSMY&9I)nFlrom$f*f$`o{#3GdkZg7W%U{PDEo@&7kl(*K)^abNGx>6Q$Z=JR{*7p^x$_x9Ied$vn9|F~UiC>PzFA9cv;wwV|M)=>sF6E+NHUCM!*dD2I(A_Y^rZThmz}wzRg{Vx7nkoZN<GwnBdn>R@R7$3Kyfse5T20WX=@x8>6t|Faub1aVkL-AXtazw~c%hVdPQN+vEV_?yZ;<^2*A{U-B}G$($znKR!#|`_g`McOq=U%b8ax-PmES4?sPq6i+ANlisp};Kec-I*Uuq}_hWU8-nub@C?t}u&zC$7DSK4l)r&kH%fITyZ_3+Xlzmrf!0z1H}?hpzZ2&agUw#>mp+5`O;I9E)^5OnvjAGM{Ug#as>pWZZYw!A<JF54Y$W+Y_!kd*bwINguGv&AnrCLK7;pN!6_<&!zePc@Q2Mew)pY&>|+yS382p+>Y%Io(?qoDm!dre*a$x$Hf9bMy4u-o+KXmHy#hz0beIYgUx;q~EVnva58W-|Cd{D&4!b;VMIBw-68oCHX-^sTT#)K_wz9rpE$;9GxMH{7208r)(=bs3nACidlY%MkL)-D=X5opn{><y#|fkUaFBZB*F*FEEE`dJS4N2b`vX*8H=4_dD^u>p1M-YoglI%*!Q7d)qC)G<IlIvOJ>|*N08Il7Gy^Gk^Y>B#d~$|mtZz5S{us%fevuD@2D61&GA{Cmy8fLA*qfm+4Ph=MbUmuI#LEbSPJZxgm6s?%MPdLh{LIEz|<Tv&q@B>X*JE^Ui;N#yj^Ma!wrrt-%}WlTL)*oSLly$QpqRouPoa-WcARNc2GH$v50Zr`47F-6>fE*2AIt3@ps*e7g&TFb_oP8L8>8|eEDNg81EjPS^wjTcl9kjMxjjH;X76h>IZKEx_<f&NzSNE!jEI1Bd7TdvYSi(7mU<ok{LZv)1VfNLH43tGZ>7FxG?C1cyGzR;?{nIL!Ec+0W1>NB<<aecZZ+7<9+caOg&XdCFoZgf29klWHM>%N{_1JE>cd4!pX`hU#hY+(C!|~(*tuiV@fGoQ}uiPb>y&``YmN)0~&O!n_|%EcWYQ~+J^YFx1>v%dTrdQTVngfyY!we6>;isC@Ho~nbdp2wr)tZ7)_3hi{6P|v@)6=MGM5tS5&=LKCPMODs)8Zmvix?n>@woR@CjijVOJsZTI_C%qUO8!1(>_UcM4g*jU{_T(U5rC~f++-YiC#ZWjBq+mIznp4dWByocjae=}646|f~#s|95;hPx9gC~fXe+WbRu_%bL8wlLj4s<~P=-L0tQ43&HzU=1;pF!y8bfk;WT{OCUhq6wI|Hk*h+_bfZoVj-E<f}pc|nnG4U)oi%iLSUi^!+yanU50~LA*!9?!Nc7M5~>~(2PB<tf))ImP^EgO)>eX_TXy)xbKSo_<I4{q2btJvgh_IkaA*W(kWj~*FvplM#}v0%G+x}>mHb>)-t)VQT67qt>a9x{c$x^>+Ugm8W>XWlF)vSyzWl{%_q{)ejA|FQ;?epd+4g*WAZ!yppogcul`mc(uT9^<IL!#0W@-n0XGa*~S}|)Bz<}k_Y?iZUo08GsyB|F8u72TtbhW#B+ZGjFI_?UISf!CK!!Fn2Vf0MT7(SLv5)Swmy6^mP{loh}mb_aJSt=s$^S6LvtzY{y=3Zc7eC$z&toxq4&@+@V2TrBkuWwV=sNC^BT+vu9#T)cLeG#=hRAuA*SL-X!Z9~_c4f*8-akbSq4fV(**V|MPB@P8<+&fCV^}xr7{%<;!CI&;Qxy`vH!_je@SZwwd$V{!(Ss-mU22dgG-m!(KB_2sSERqfjT^fOIt`@}SWpiXd0?kHY46E<LvN94CHU=}8oZ>&}OC6Ir1O2BXPUC9B&<@S*WKWZ5-HXF{b2CP@)ZDIKvNR2cAf^gq{6OUZVSprG|7JVL<a=$~AOj3(t8GG+M~1}EG*(u7?sKCI3RZUYGT!hwx8g*owZqSg+OFjh)m`fcXW-~~j1DZ1mhluQ>=gJ+y73oFaH=Mt4dLf~v&0uus08>|s0)NZCbtozFayZcMdED{3&QZ}CM;zX$L7u_Sfn<<7|$l=Hc+g}<EQlCq=Vj>uQ5Cmg?3`Hbobgb@77(uH2lTN@7EsPQ}2@n`N9i&QE*wFa+<6?`h5M($K9{L>^|f!@gah|>2op6JTA6v^ZhM`nYRTLX|iJksC)@h0<7;)DeJ2@0(Ktn>=_w&++pF!BX3j~UuN;l{k?&#oV@vAJ;DbP6zBTHWP4ck{t%g1oD!~_uFd!{%5!0{Ila$5_TK)~L}`7CsGL)w>}(k+l7-;bc$s0F4#lt!W9pdi&VVT2y|wIp{w34T_A`P!?$MZ5mJnmI#p8>I<i-sd7Xt<03W%>jjl0(#W40QK`)+L!kS%_Zj~Y!b+}h?}a}+DdzS^%ofL~nf>%Le&ZaQK#s!`Z_d8?1y0MwVbSlcuD$YOsIY`|G&F~U=ZHBfa<tc6-)ZKC_YEQUM4g!|(x>drXGNhpwm(9-GmwEe#PY!H)nv^g0Sft!0oCPM`s)7ep>ACWBtUN=f@)=nviZ0IZUSUWw=ZdA)aux8dl?-<UOurOBQ8mWU1ww(?QIwLfW%vrAAy}Nnt()!c;TQQQpSgeMiEmyaZE0t8UTReoj_0neMLKOGmcR`{~IFr+1^~{?oeH>9D>IE;D^m0fea!=l96l+BnLVO9Nd-rztBRKfW9P9<+(0dApcOKp26GrYml1?SZ*zGCBYedu_PmgSJB$F@<=4nwQ6ILTLRbw7@je50GTH5}<@S51p<He>c6<UYr4V#h8V~w>;)?8?$9je$1;R1Gmy*Ikm`=Co|g(7}Fw4cxs>P~3Fk3%#&BK<n1Y9w%Al+<f?Da=u)DSh$ka7e4Sl5TCW(xe28`wqVP+Tqs<J6}6`aPNV=M_EGisexGvBR#;5MoPpaiW!7o;156!Tb0tTdTq8c7sGN2i;Irtc~N8UfmaSD#4C0)GH(fw`|}tGo~`1X8*foQH&&nJF=)2(lcGe#-NmktwBcm8@NcXR#UKjSd8JZyMGas898A8QB?%QV7C+@R11?KBQwIg%8)Qe=K^`s&1LO04L6+D@WiiS@R}k~zIagr~p5N}i@nGZf_uiH34i>0tT%Je~i}@bnN?&#iV#hHS3F+(G-WxZ0mdVZ2PdC1JgZYN2u^SvLrR0R(`G-m$SdOjlJoV0f3WeNack`|8=`-HhZ=KCI|FH4=2E2grc*{4uFWy{x`kvZ7R%Ks6ed5hGWfH_6p0I1aAXT$QbfRKe5m5g}g_v^i{8jJmPgps_CM4#KW<^B7CV{NeqI}9S77D$OVeAWLjd56|X#%c)X66SDqRb?8`C`PhML6TVnlXw=9u<iOFivtVGEnm55dAqceJar-_%hZFaP6>mN^s@LAw|X>nx>3));4-<IM!4zgRdDumh~mjP1+I5@Ve&jX)c8#Ci8VreNMKx7XGKm<H5Qb++3PJxyaLeGY%F8Tq3+wS>dua*eE)xXTSH(t!zBFV|9x$6+J$rDEdLjC6=57nkJ7es(5RN5jHp}uu^e%a;Q><IprcU%@YPkA#ea0B`${Ip(QH=v@#58Q6CFz5$eFUaYAyes|d~4w+A*K(sI_A*xaDJ7%Dc)T!vdj<wH!aq9i|2RHe~_sL1pmtg3hltIo0+Q(g9FSjA$AJW8-4`_r~3lnc{Zz#vi?F(<P-PuCv5<6XThlghGsz<s!t4EP4x%7_iDt(;XQH03>8$To^ul2L=GqBwQ+I$OZkeAibc>P@ef@D;~Tb{i9k*z=>Al=MttK2)<6iv8pxcq>_IEe>(?ky?$~Vv?O$_zN!N<-_H>n0m>-R<4lIS<eg&Vh{``t;Pe;Qk!Iy7jsS5EyFRjS)V~YY{IybRI0nPv-ynnqEv_Zw8$jIUi<B|JX2yLfw)95JK}mVzn8#|g#+<E8UZS@J1HmGjvJD^%d93k7uOm5W$+sZGI3gaI?!2|0gSFI=2`56uIsXX{PLmccw`UEyI*h3D0;A^P3%!$hp^;ODE7NGq{Tu_hR~}SlEb#|{FqKSup{C*)asBbll8@=S$O3zoiYCs<zGPY5(vTC^ZVUvkAoa@3a<}2F9CHL)G<Q?^^v1{4ml^=r!)pCx)?NpA+Q_~Brl1$C3UN2A!HP=WoQBYQhtbof*5S?gd^5Pw;!<!v^*Pj0*G020k6NgWfmGf;c%Bgq!46};U9=w%}^YWPFeugupMprYt~H28xOu%U*$2_nA~H2M=k^`*W}LGwg?{%OEOJ&VG(K}dsl1!E50m##^nCCt=7paJjj^7`*DtH__0&`D5f>H3J7MM$Sv4oBWpLl%fXA@3f>=)86sHBf0z(39x^E;wTVk3rb&VQ%@ab40aAhzJFU*bLb18jKSf5>kPPUi!y4(F32z&TJS?*t{5XRB!bFqw7kKpyHNAxbvlSI6cW|K~ZABK5L)99Ii))Sy5yfB36k9IW->_FqMKe<pYMGtC+?BvM>g^_Mb0&+a*93nF?K$^{nsLi5L1tctv;X>gchJX%eE(IuhkniOU(IhxpTUx_I4ZO&!orAT;LIIv6h!F=oz_}yErKy86`X+OC@?!F$3wd-E1VAh0OElP<1qbNPsN<F1_J6nSXE%#HFPn*vJL)BW5Rr^Sf(My?124kJbxFBq~}(=EAN^v3@;K=J5=fm5%<D{y)bXeNPBST=b2%LOzOe&n|I{_04Ed<ws9*CpXz_>AKveNdM>O3Wro~J9_)dw5?dMl)ZYE#&GpOYet2^J<unGX^Pap1$h~><O83@EclpcS?vtUgp_oS@?*gHFw40skqft|tD^ygT^yGRj=4vej_KmvDVKYWyfr{1}w=Zt~@jC;#ZL1O3$ijAj{J@?n)2}>3Lsjx~XfL=Shn5!@6UZ(G3s)|qMXyU}Oy!8%XnYExso<^m(0KL_*_H?Q>=Cb1k7}9cNj{BO!3n#ECJ&~Nk8*+RaX$@XTk~<h>_gsSu2)xuw+N}kAWV0G0`@^sLGxjM0e<B3nuzqjnh+m24ip;9Z>`@mfs#2^?ICAo{``Nw9cKF!gJ2R#)QAcPx)Oc5z7Ug9pj6ss(U1{lGX!So&cimo-10{Fn`+CA?w60NG+(cXU@W0KD}YQK;<VMn93IfPf+jjLm5_NSSV#X3#ns>=4DoKzD+g$7z-k@LUMi5umqMZ}bb1I5Guj5}b%v>>8XSSv3MKr8kVm9s5E>7IB4O;Ki#bt=PkhNHRHu~tN#qaJ#m+8@m={EX+?x5cS+!7MHB5Dst=>;3<FH-JtaJTIG0ef8dA;`~wn+J8?a2pHDADVyP+7ls!FzUwbL5XsZ=SnkOeO%I?mKt9PrvP+{^R=7n`R5ETtiThK>3GDGr`D*G!dxgKbVr3F=-sKQM84y<NM(l=ulFJ*<#7vN~$~?`_j`?`G3KFo>uKTVLVD<Hc|Y^^qCCd4_3bLKZA59wm`f4Uh7?Tpv=|!T&1*ybq)ekm9F)$pS`1^@0zdITdo=DcmW9WLa{1iyq6tv$^pD?S;THtd*_~LRO`*6F7a99j9>JUGdNH!a)3c0bcI@^<S&3TZN{ZXRjO6s?f;;FpxM$v!I(zwMUMb7*G@3``>_0vtWb2QVg@g+9``F4;p=7kz#n&tV84FEsUgub!M!raOp?2dcQ5~L^Mg+qvkILfldGyOE<WlGt!{6I3Y<G_<MlP3lHJjK(L?S4THHt7MeZ;{)^_=6-^twWW4@?5nL|sU$K<6vkkI_i4Xucod%$fj!fsN70AZj)1cEvEc*02sfDU^c2$0KS6+>emlG_7)M=sb%Qj-thptAAw!TQP<-H*_dCv@c$?%Hr`#VVgcdmbI9V*Y4evW#8({wREZd6lSI=*O5hBH!H>O!v1+@ZC!RF=CqzywD~AnPQt|)_okQ{lUNNlc}&%HJ_y4=tmu_{S!<J&Io`}Y+?F8xYTRaCB^gRh&;ClJ?cTMfVUPaWeg{WG3rF9X!;gz>p9?NfxvOeA9?06jyX7vfAUk#|G=Zk+Ugx%;B>D)_MTjkXBF?^6(H~4`S&d+mVd@G#Xq_1Z-VSqypv18kNKeh85M?v4BE@KuJrml9w_4f=Opoyoh14=IWT%`Jvnq1tb{ncFV2-6x6!Ip>$Txld#UQGX9>sP%>TF(1x$!>E{kxlb5E7NSW~h6Jw^W)lWYxf!EgAoKW~e)pu$A92@@(bumrF3xK-`L`F!M;jV{CTKn^<XiaSjH2|!tLe9fKW0?1I<|1cZ9jqF>o#P}?;#5b#GmbX~q`3ts-k*w8av}7hV?4&ol*$Z15qDrd-;}8@W_oDCv6h&y*?lY*ZX}Z<8Sm5ST(3<!oIHW%&1_eGMHU}oVAU6o)C^6~1F@#BJ%I_6Yyb+pi{hvbruZP^`cvGLjLlleZ>(&(H;ZkU=v(Q*_iY=$6<@!G8H``MS!>l7qt+u-`OxIzY*~$sGJY0sOOAYz1qkP!mA7-s$JLiC`&XFRBPQAWSq2<TD?<=~pDnuGVie%tn4g2&0eUnQ;B*zsFu@i|U%t))K7pTad{RU=^8M2H%UG;)p(VzINTLcW#sP&+<z?Fd2y#x~~)ffCnnNo-=0$$=rEAgn&D|YA-C-Rq^;v#gz?s$_P<|U$wdI{E1;zosMu%d~M2cp};g7bGKSY9&VE^^UoeQMbtYs&ua&wk3VR7wi`0vIOBktyQaK<_x3wpdW{QoAm))Z>49F$GLeFjdpAHHvsqvgQ2NZPtey#d6s#C!=(j0F(d&4-==1h*Z80KZ{z}fAg)ohVnE%yc&u>W|E@@(NA6h({fN)SJoSdB&7~)!P!a^&in!R868D>QfO+wnO4$0(WZcerYbcaQB~Qmw)z%xvC@1pmBf^E>Oa%*t*%}sX{1ZsN^||bQKZZkazP(f6WtS0jzJM9V-bDAf+6)4(1lf0?wC}{?hAIKk~}V7M$45dQ4C!BrSl+D+H=quSQd7fnL~}M?UiJ;Zmn6zjBNvy4+2GGUA3=;UBf1rk2fT^ghtPm5ScFkdwyx~#o;cs$p(JeLYd6ry>xy!NB}JhEQ-mR0_$S%BO{Ff6S%~!K9wm2uZjEh6jl|Wu>K878Ed6wSrwYUssy(MSHk>neMVpjuxQI7vsL=ry{r6!0aO2PJiqF_{oxN!&aXYXwD#<yjc>2>$a~-arF-iF^U(}A+*ly{5}Q!3U|le$`C_ZshDCK3QQ2<=Kb%TKCTgwj2RBYy-^aH&5>*!IV<bv->qwOMfstsE6eC02V(}yPDBT*MK+3?5O05^{$CTWxW&@^5s5O!cZm|}#LqBc>zJ1>zi*#f<><D|Az~kqC0z5W@#@5VJ{ubs@`3FwJ5PY`=Vim1=jZOVX-yrTA-{i6C1$$71ps@KX>^h!M;w+JIl#!``3Da+_azNP}(}CaJT)x(`wLx832ZKBk80N*0mzN_tQFppB1m13WwP#a7R(azvQXcHY*vJXp2>kdP@1t{UD&B=Zu06fMZ0dGxS!g$WEbE`#V_nI-<!q0#4Qk~h#vZVaj>g4e6x&9p%LQD60vo2-Y!;VdfL#fv++J$9c_<@@j$|~|SG!rPwJ?5$+bjSc;w6u|uB(9OlAm^qo|3g1DKCWS4d^Y}oAJ&>wNi6iBbXX|0pvBRU8ok9(72~uqj-fM#P^6VGA&5tt^wJXA1jJgcJQRIA(3sETE%ZCn#F3h<o<fOQ<bM%i>{+wKm}7ZftGbtZ06G`Eqq?9)4=#3N!d)j%0<4-1S>qWwqm>w*{fQ&-jbDz>_%itel(u!)LOskxbAP=SURo-%U+a?+!KwM$y)W!<Ww?oXPXlnX7AC|&i(@W)2cUPFu@d1$9UY?ZW*UGy~!M$(sH@Xv>ZhXxCc&U8;&hdH?7qz0Yq$dNkIYh6qq;Q<4k9EmLj1vT=b4;xC$&%aV;VuU@+cQ<STa~u4#Bq<eb<q6VB0uvpbl;@Px3}vvKB}6G_xl0I;X~Ldv5qqBWo-$7x5w={|5%G(YO^L)k&hx>aoVi4N}@o^4iuiBy-IwipoCb~$oeGb`W>zZ}?VcghvYT+jO0DPBlzqA$G<J3=4nP%`DZ??E~2Lx2z_#}fK(cV0dG8vLuw%Wol=CTWn7wL?XK#*et=q$*rnz$?`7YeMUX`mTIBGoG+rR^`f4^M-1t<q8FE{%Et~TH8v98;wPIfe7%_fI@2#pP@a*#;f5u_rKuJxn_s|rB&3w2vAjB1v-WKO1bRTLgr?HzT?z5LC&jb&T55uw^(jzm;GmFsoP9;%sTC&zl!F}l~;KT98s*v39|>9CN23u(61ptSc9vpEvm)>VVC<l(LlpwYrfbpwX8>>=26pzqV$gj6y-6XgH@0Oswz*Wokhh5C$Hm&QC9=53eP^YVunidl_q+c?J<YcvoT1~%3Q2z#O0)cpB{06G}kbUf<=&r_xmb%w<N|F$5_FHy$Ry!0YKkaAC(Wg`(BI1gWiS_8pz-z_Lgx9?WI`UQnONPblPZ^{xz!{IioGEQzq%t(fvqAx3*9QYUL$mvCUXkCW0NDxC%3S-$bhV?HWjyEms-=6IpFug_~t<Ifd(|>P)c>@+N7-1-TAA=!MNqBfhpdg5aWTcZ+bT4N^*vN4YbjrVwP^FoMjWr3wP<E$srE+ad#UFPPMju;)2_xB2l_77DYjTjnBEo>Bd+1DC!9<BkM>UkDQffo1j@=xjsjEC@a)7+y3s1gJx^SZTS=>!`cl(`<q`Haj)&TJ2b^e!QmRP0l%~J)0pS`q=&>N#|gv4Lid5H&~`EG-XCWY8s?WIwD%hQ{)E*omUuA^iek`g#%s8ae4nJoiE5ENiB%5EK*}GwQ~X+@9liV(Z%9wWHM@Ftxw1VcO2p1n-tI9a&WMqifrz)2<kz{0yainTt=H+$Y}edyC)J(L401{O|cU}f*00v6xEZnQz#$xZLJhJ1_1=+1Q0fTUA&Z%qcJv&W`2UH2T25u+vXeSib<ZfuJ8;voSuGR!cbI?HYd%mo0LIKuBL3_*gVtGPymr;)$VECfWv)p$w`l8Ml#uv)M(~kfAbsEyC^Bzv%{(EaC&5_hT%#N72ESUXZveAU){fJyWg{srTx&MgN}>#ItDE~F=FEO!~0a@zUq73<9NJxfjfOf*3(r@EDi?@w_2GITBAdBIhEd2>vNrE$5G<Py3MIjRg~=D(7|5XsztYiIx*=&-dR{!dfhGUL01U)je;m8MwcJ!v@6wCE4ffS=Bf?FbAL)Lm>BD*N7cgz58^=u>&FTdFs@LDV>q~ab<u^>Qlr?UrP+UQ_nv))-Fpv1WoomXyGmgo$f?@CgRhz|0fa3PmAA)e(Y?9XT1Ljj5u9W=>^~*u-X6uBq(s4h`)rL)SBt1r<Dr~I;5{fAOGzszi_WL4vZ{d>YiLLwGK8P_!;mQel)$Q*c@E`-Def)8k+$rf$fpv0`soK_^F_wz^DodTfd!fB0hRzhP~7gjeiuekDGW2n&*p27E^pkwx4wFM?fFODi*KwwTHZMKS91cib!6CLL&)RAcRId>b%uk_zIw52Y_Pr>!+s@!6G;UO#z%Za>RF6YRlqG>6Wt~o$EreVKp-^RS{F(8M5ToR-O(bd4#%PBjJ0|j$YG~e{+X|nVJM{5O;%wtF|}LfjO|0@Z#O^v;n5YI)V}-vcbIkg{!I+2eC8Y3Ymyd>xf8uYQM2V6>X?OQ^I!H2?hy{|1^<lMvsKx}`(RRe%N}DGVzpeH8eo@{NUICaMutgdaf7Ojj>7ur<5A1j<kCa(x0QymVt!j~y?{Zn6NGJ<#MuIP09zkrpUdvl43@RUJUYOs$t!WrdG<>QUMXUNyUk>UlJE}D;hp?eRBt%^s#{JvM=<f0!eEmMRQg!G(WG;@Q&Zaa*^)WL1;U47AUP9<+(xAZQ}8iNnoB7k4aMD6eMyebi_Kw#rDG1hOgLqW*=V}x^USSME(<G2c6vH3#x?2R4?h?c;7}KE^`XC%=Uq&urq$Snau%wmNFTt@0`T2drz-rXdp6;S(dv{+ZmWe-;vPk^;xX?0n0zW8$v}BHTo%(#4Et1AgFS5ee9{=YP=iM<IWnutp9slPwcc`hRJuqOf#$$+f*2n6!8#%SSV~Y@xD)Kx(iF&Ihf;}DA`3qgi7Y**6KQy!NbE@Lpyx~?!=ESUc_cBCNK^GhnhHh}qlt0$oT7p({5b{%6THG0JdINi>^<}h&(L3Dgr3LYDGR-h@h@V_aTvvpM23whg&!Qlf5v$iX&7ai8tq7ov64&z{y)x|@=7q!5gN%j?>Rl1fdB8H!SdO{sIyR##@^X*((t7*8ZHc*mB4!(a)yS$ir_2wp~R>JOd94p!A2@TlSw5q2&%M%MJAod(6<n%Y?v8<a0Y=v#jI!~0so(XN@JqZXd*L8Q^T%!SvHYDNRJ^rS;<%;GsdCMR}X-Q@ZJHXybjC*q3^>x10O<&P%XB`0E?7ZWd`4obOQbl@R!Rnl1Ypp#K+`JMgTWx7FirAD;iC}|Je%eNQ|;FShp-qctYURSYianJnDZA2DpP3LBMikJTbzs2tmw>pbskWaS;}TzLl1JkERo&X*PcjfzeE26v6MW%IHX9bcEJPEaoU+^(aAhf`4o@n;6Zq70+@^868WEj?w0jB{0Vwd;xf~@x<sjLtO#gQJClijYmKv3m{9;M8<iuEX*mL*rBXR7KTg+I3iadJCcC^k8|*{foEZOqqLy~OtOIMS!0L7sK;oJWkoai3?jUb6<OGn;{u%-;Ih!ggbv*?c#l;OHyH@V(upxz5J^GC0F5)WeZ*!RgU^hxxxhbcOk)7<QHDISApkPVceMB(p!*mbmFx`uK$wsE`)3TmIL=TGN6X4!MJ9;Sh&A2;?-9lEhfQz?44P2_tjLOB)C^h1_+ssV1sGu@gim<~5CH;!MBg3o6&(4PT&o@MCxRT;hL^xJcd!wTiy}aYaUwhdGj;%ePw)knQyGUADI%(4f`kI*q-knnVvWZE<e3EPjaPzMjcC+p97a7#Q<1b7)|$~2F}ZR0&KQALEXz2M8-|%;myg5N8t0SaUk8xl*38I$CIHMd2Or!MFp@OGCqd^XfD$q?F2FqjLncZg`Nsqdc~oK#gWLp6Axq>*Q1%JfgE;Ptk318w@&qBt>?Yt#<7`lpv`@gdCW!b-fTvUVK1HJv8#|rCp{My+Iaf$wqYO0`&`PIpm?MmT`Gq)vQGG9^Q@BC1d|!$$W4kd%2mRq7)^N<3kl)6MjMLWlKZCe4LHk<l^fY!O5lUcQnmB4&pd+6mGSzg3O<HhA<f`cr&OijGMaG&QB`g_}EE-0f&I*dn2Z_uzJ;sQ!{4`WZlQ@7F%*$Xh>2VDa(#T!Yj4#W{0DHxsF<N*&LS(R+G*RJkSr5g7j9|Q?B1|Pi1k{Hja9RA}>{zshsbq+7_@D(wOFt>vEdmh0Y}3SWGm?;^sULrsSi|QG3>SX@rExY~T!Fcar1V{!2Bw?lj2CGmFNB$lWVnF9Cxe`Kgrk|*-)ZE%Bcq%}ipDUR5ylXt%m>^Te;6|pUxrpAJ2)#5^<XL^#L4hOybQiJB50Ic7-YkvDdOpZmZgCY;|~+t_^^Q!<IfnAV|W=%Wt2!T>y?+H@HWb2M^O`MjAm*11$R#aOU9oa3R_MCQ^p?>>0}wqg|TKrdtL|=8D*SOvVb)3<}~qVg~g<SJ>yT74TzV)WU?gLDdHrsX!>C*%RvD=I!%09l~=PUjAm(Tp`5`BVLI6{VzFalUjnPfAJ$rwK|k4X;t75k%m$fu0&#`+2kaVuRQ{C)hD|>x+{pC@o{c|a>UKk}JvPF&ftWOM?Xgk8ro~)<apTVpffu|KWozs)<4gCzyzys(b|xP<a__Nm+D1NjfPLc+KM3%l0|&<+w&8ed<l)H5*?RHTz{T-roTkX91AH8RvZ|br23C$g6N;dh24;>w49~=PfS=<}meBxFh-U>Rj*JVC15d}FapH5LN0`hw;g~-ccutR}*}fD*gy-})6SF1ff#>x22<<|_Z}B-j&PF{ZAPdjwab42J=QQ#25ufnDb9#ISp_mT_C`gYpsnMVEI39Z@m@MlT;=zXwXg<9~RzD#n5?%&VnaB{$ljk^iPEU+zf+s$wCq}8Ir2WY7CzwPfH#$70C&q}lNo0iQH1T+4N#Hp>!8f{sp;$CA!Os|?IbdOmi;ZH(U_pi${fN)VuppDlh(b{lj&i9Hj*Q}z1_hbaC{0^XJ$%ljgq$Vn!8T2eaXDEwhQ&<n;CzrD*Wfurj6Ulh{NOp0njq4lG=|v-HZM327G%<CE(r4;(|9V)aAXu^FqiZQO>j)Mg{h?3u}B<C@HvytGT~-S7Q$T8V>BE=Pw+XD-ofY00S3>RG?&RacH?s<ElDt+9~5LV?1&^#8J{!6=f{<kGaj%rOk@#UD?=x2VvYWn@ra!v-lQC;@sv$`KI0!Uu^^KXB8#X9b0IcAp&a9(S!SG2PZEDD$Ydr2LWx3{3_D*dLI##(h|`ZMSc)YXV)dhvtmAWLgvl-bmqt)$Fv^&tEQ9%sWW|YG^a%4Ycs)L6Ms{#2!=VPx8RGS61#zJuGcv(R8DATC&Wxt``G<oSo-?Cdob<N>lw?LTOd%r{44yNigk#F?f`SY`f6G-IMHNVv3F{I|p&&C#OkL6*e9nySko%nj4N5YM-TRyZ3o@e<oRNxe!*sIzq#_p=3Nl%y7!x!JpEFs;QRLFVb0#}No88ArD9B_-1!nN+KtYDi<-W*_&*;HPBbQTzf=pKMQMpT?B$H*z7rC|JIg=FvGS@f3b7oA4W>QTAL*s{Zi5BpjA$G405(+Y7Y&zoDiq9E5tTR%>%V08NS-$Hys=#w*jL5Y@lEzR&wSzMdJ{5S*j4_3vzg$p|8Jp0=6nxI?NQounbAf`)jx<wwiMhaYW(N};{I!OH3?1ARE{6q~9is#ZAN!ynL&x_C$r<oDL(D!UPrdk@5l46l+a0KJ<H`pi?qNY@#{}OZqBYEBoKt6s0Z@<`PqXC`vH})l#xoqrg%%HHGd@D1om@hA&ft-rI1OJxc+Sv?URf=8&WtmWQZhMs&d{kor3?n9GR~GtvWjt3A5KUpikhfGq;rDb8lE!~X}<bmDo~Ii86e{e5Ll3z7~wi&zKP*EGr^>Nskwyb%tV%x4?aJ5&P<GP<(6y=lbPUz%kL3BI>FG)*L=ckCMGz<ICS87B*oy*$(hu+Yw<ybhK7DB4-e`*%*8{R7sDJpXdwS|-W}}HQqp>9Ni>PNQSlFT5{d4oRIFBKilt-nUevC6w{)yguhiPR_$%JU+|oj=*s63ElE<5P$JW0}g_m9TRTe7k{ykFES}u23H6gd}$kF{K$as_1%t(R$lnX?jhJ_a9nd($s9R~hH{yQ9DB0ODs_{hk%$%SL(N;5`==0g1#b82<i2U@~tDLd<{_W%E!Y_b6-o2>6-1G6tKwA9S(+(ibXE9qV`Qv=~49ig)32osFdB)niw{WG}tRQh%A!5|8~9DJZoZx<h@A0HoQP|c{zis`B;1@?4+hU!tVZ7x6LERe9r1+X}JBY&y*sU8{9D1JhOYHX<_HDjUWaS4P4o2=j%=qV%e$@S-1^a4&rL6@GG<`|;hnL<yT`b^98byqxe%vYsqr9n;7DLpF%mh&dYtGVLBLNT98PD~hnC;fT*Ah+MrzW&L*eOS2Q!ZDcK94;dHfGE&%+l3-@m^aF`t9z2m7udDQyq#GkoWUOaUyrVopTv`p1OJ#CaB)_77w~Y4TRe+m>WyBUaeiuGf3?9Q9S=mh7gd3WQm_IH+L||bYAovxkH)pTdbA(%qp688rI_8%pldQtHl5KLxQ%*g9<k1fSXv?M_2R-zncidrD2lrn-ej`cd{}I3C{L+(3KwTZv05AV^<1-9Ccix6orWrT?Mdy|ZO+tNu6BP>bI?*FEz?DVnL2DNq+xjssUrjfWcuwH=}xmnM=T-S^y>{&lJZ?a$bNu0L8(u@<XT)rj^X|$PdBsG#iLg!MdzjXp)NQ|292gWTR8z(*eNa)8jZTVYANO^ycCt;*rd0gen9-gfdTdBgmsjkUHWXpB1j-J7hHsmJ!SXbV;*|qe3VT3{#ne4jkan`4dtzMpEsVh7~u?_`AG{ZBm%23y>;M>C7mO%3=TO_Zpp=y%?;Nu7UhpAVL@(qVYMwzw|1{Hi<8{cz)^WL)jZ_jyv)uZmT0Z~MX_DrC!`lUOpgwy(!=SrlS${`|Dzc=1{7QEJ>?2;j#g9;q8iIK?jOsY^k^=-BR8s_&v#aug`>q26(H3jQ}1?~fj$9#2De2F<z@0<h~@AgU!Y)zCkSzT6o)ms0^Kr?mHJyimW}1p8#mrvdvwXWdU5^2`R=W+11HfE2CL@n_Q-Q3yif98M6i$g3I48Bukzsk@?{xs3|pi3Q{4U|od(IsuVC1cgmbjh!1Lv9ikjO)Sun`MPx3^o&c?UjufO^DhbI@;SI@&ASi!q+$$PfUCaI_OTw7V8fHZ?n_raC5M_;LLDddh;$A7u@=pJvt0^49>T}pn9)u^a@VcENNSBswSY&?AmBV_%-09mA$&E-$Ll`r7YQA5gLKRk9Yo*M9$(x9_`;j#DZjqa7J-j(a#xu?*JxEi^ghb?wCo_`2^dS|~`zjqy*Y&?Ciw)!2WQ9t)p_x#n(_nvR>e)!k!^@rW%FE{>rcYF8xIe3pn*B*OUFT*g_9<6$B-`zZS2`YJSe+pxULf-A4mp4z}W$m}~_-R)jz>6QAeBwv8Ew+Ef5IcrA!vEGISsXpG3sK48FX`ma0-UJ2&aXQ)Hf@Se-mbO^m2xl$h9!gxP@5cfN;eM==>ExpC<DSxU%joidkHvKrR40bp(JVKM51e3JZvHLjz=lVUV}?)q%}sdhl=RP><n%$Uw{?E6t|DQ)Z!ocy^^qudn7i=LgvqbsEmWH6j{=0AfnUGHl=GiHQ{ArzbNjQkp)VAI0X0r-}cG<ooc)COQ7G)rPydZ7-d$EqkxzFSUcF^v1<Kz-hpH6d}VH)^Vft!krwhXb#N022LxFbkO(iTHh}52#{yr2BWhdmKUddd6n5EHfLyR&a2GzkwEpxy-`x(|+m2ejduzqJrPN$|^f|QmKDvrIfz^r|Prra#Y}+}o2fNgYKdZTj<lc99dj<D`%Y)O=W6^9pzvg}Tv3LF>@5^@>b)cPeWAofa?~QXC_n-09-1KVm^wYIRD=Gy108cEeP;4SGDk7lyBf){#2f+4=_wXzykZY@#*H(Zk{?&<P6HYo42S|L~z4RG;cJ0wJm8TPa7I)j-+nRg4AT|LV;ncd;S~5J9B{-yy{0YZ$S6zg%BD^uKbE66naV{1cMuYBXJQ(apjzWztLj{8Ia>7AnD3ejntPIn)uuBM{W)Q61kKgiESEwnbHTm?3_vA`MS`+oNM$k>EHQ}l38*Uw#-_Uf<8HBY}BO~+Z0|8hm;do_4V%g1PWjg8ONFacoHZl>`QRxtxPo}0{s99QVW<o&Xh4vz%f&N9Y{G*BCble#X0!az$11BO{OAL1#273`bD`SSi6s>*0zP59J1XKq%Ke_?CdF}b_?i&w+UD5DpnIr%v(M}BiGl8sBpG>iUT4FHTB^R@9CZ|<RPunNN!W;9IQ`)O@0XZOIE<BxBUlYN)7}HuA7|e(TWak>rkVa{((THpmOyMP)=&%~aCas3+mP4wKXw|b=eHLSDSwCO}9d;8B$2bWg?S%?x%kHSu@Cuy1Jrc-xS%BrdvSbMdP8h9xDss0iLK&RzjSlq@MekSph{olJRDvmnWW52ecA&B|#>3yf3k=%3_;~ZorS9F^-pB92o?^o2*&CQPt~3v<{nF;G<&7^$6b9M;@@n_;9hCS%GQE6fbM=Fb`=7%f-uZ_<<>%zRKs~6GU)}LK59pgv8`&5El*L-fwGkZUBVn)cGrv|o!>bQX=WOv@*r$BzBx_Ipno%e};ax0Dk#B3-JCnTDR^>K<oSZ!k2Uf^QuEGl~^;JlU>c2MY0TuhyJ9Edo|9JD-4Mec>-#MEf{H1&EGraZu?5xB#Ko=%6g3xDJXxs%1s^5Y$oLi1DUV$NuweiN|iDT|ko+(tFV>x{`lgCW5C<aeq|M3d}sxIzd>R6ei;+Wsu==WGKIyk$x$-4VlK9G5UwR%Zjp^qNrv1wbd`>l@D3JcLrqzO_R2FQrs#@C$jQ!^e?(pt(jJkaQppSZTFcNGs@Y*pysNhJ+T1wY~#$<k;e$kb6LsW2i3Px@0#ptYsM())slG3@}4D-~;4)PW)rJJ4!3G}pu%&qKECX=0p{?0eIc!bzJO>`yUm_|sU)N+6!Pmf*}M**;iRn$LiWB`O3FXXI%>iU>N;;N0y{Dm~!8Lp+hd&@@jeU`xGVt2YU~O#7Jyh8Bl%Hgp!*1Ud~#p`1J>bBOBI+RQ?b{K-yQn@op6*l(ypmz|3Zq-BNioTZoHw6l%BH;;+zE;wg<APWI&Mm-5B+9pW_7m;yLmkLOcz~Uzvcg%aA>NPcrp*7(dYYYyomI~%gY+Kof3KH6pEO3~R!|;Kt4l3b9j>CFJggV+|+^Xggw5{)B4<*U7oizBV#5p;HLL5k}#N>x^AR-P4p;Rg2xa5liSS(5juA=l*w8WNB@~!AcYqwz}3<E?mvCL5P&eZEw@o%q`mrU=t@%;nu${#FBH+xD;0YZcR%a?W*U|(-@nZ#bi9Bs``t%P}}SO|lbjA0-&k7^tP7Jp^^@&oU)(`zfAudVz}i-<)XAPGZ%0)vK;QnU^j6dO8u3Y|KJl1y0{8zO%>R00phVY!bFB^=@Nl^^P~XNSk3q~;z+gL8gpD2{f7Sqp_c-mJ6<^Pux2dx&9x>`D<;`uQREM9HlVS!kc|QxT-zA|`B=*7u;=&X!5Gy?ZX+^<*m{iM~Y_Vr#z`Gl|<fW-GuKqJBJv9}1~DXvBy(vUh*ZIl1MLoN}h7Fd6sB{@`&+ORX7(5CS&F@c$9)fXC`n%3^$PlBn|E$|R5!c%Q$ue)TL_la{?lU!pegy@%aP_g{#8=n{h%ks(KLsuh@et$sX`<7nIfDkqH3g^|G^uV7YTQdnE8HR3y*bZRDK-vUgAb#f8@5eDE<Gytb3a_Jvo0H&TY87G~|jZEZ5#&u(EYpE6G2R&6|KBMSrNZIn1db_(yuUCwDhzYZd9^w?D2ah02tt5Vx!&W>BO%57vY|W*O!Df$+u*EXtZfm|X^Z&JXEj)G<SNgBCjI6oCj)$tQ>gw(rjChy@B%83d*eJqiR9z2y(7bl;48-z|gt80Xh(~s{%aSN$LnII=VF}tqE1u!ccs$`>>{r#V>b`yN%-CbhMu5S$U-dY3&Z$#X=X__`kCEphSxidwr}`i}xObS}v~#lb!)zLl?`G7SbK&`jd_Cvch1z#^p|PLkj(dCMN}WdgZ0j)U-9ud_h|0TUQo4hm?SC^`>~3RuN1yVn+$+LuHzibGFAw09%k?eiQnwg550&+opEPceml`L6(h%r)@dqtjuC0A<=ePJzlP<zor3YFIy^HTO84nm!&(X@bj_#}O?f>Dko1edV@cUmL{PZ(_j^@(7_f$)O@HDI<d{&IA@qwvywn2Vs=~&sUlZB;fCDiSDr7ik;VqI=^GMyYEmfuOCtQrjF+bmvKc0Iy>b-OP}2<K#V-SyNVt;wy?)F*Amxpw9+c30PQD__sb^+(G)?qjSYP*!`-71?Zq;$bga;y4_q>Bnw!u)Py3qr0t~)VY(ZCe^u=-Fr=5<+IIg*QVf<roO%0Ij$kS<C><4{<DBvYMZ&kETO;6x0)j7*8lwW);m7~VRjs1dz<GsI{5I9&Wqa}GXM7d{a0Su|M<VY`Q7XNL?a!4*`sF|ai5)Wt=Gp&FmkQR)B4ET>Xi+A>s8b|58dSx%mFg?4oR2hqyzB@jGf2AZD$1;`-N%?g*jb@qg=m$Hfcw<1=HZ=&GS1*!QpdrM=kDgm5HOg@R>MDXyxx_CXP9oFpipq<GYhF!@81JIw>a($s*ASKbh<}<>X+ZhtiT9mg!=k@{`PWag3JUS!#@E<Ya2`xG4FqO;tzm`KX`bwn<0PZF}#OD@;bwN^$R{F!w<7v-i%NYznuZJmBgHCJzXDH!S1#6!E`&Z>7{MwJw>)nJ*=|33R;uyZ{~9sj>r<72R2efH31<P||Ir4e)G>!3@#?xLc#2EOrx9R<%5NVz~S>!Sw4ERhary`FLGRo;(_0|4i)vx{fdf!2CMy9h`q<PFBKH$22(<;JJq7-7zIgtMIYi$<F4w*Q)bv!2##mx%SqunI)#3=0gAFgRlPW;EQ+WxXc~j90{Y$yp}*Sn3zi8W!4L$b&0n<FPBxT9-9J}A{O`4P7S4_a{GG8{Z1h!_p^BLB<(n}6pNQSlab)FCn35tHQfxRb*wf9*LKSWlwFvjRms^$CKMWWL<)7ka~+&9nDeKgnj<EQgXfiPm|WK^t!9yq$A#PMamkymc%<^7w?kLD@NvKF+Bq?({r<3c=d&1Ssem23{5k4$aQKVn_du;nGwUpebtxKM-B^m$#$sm@X^IR!xa#cjs~gKNu9}ye%Wc&0<i5Jm?liCMqR>G{kA{~5&g#Bf>D2O3t3<;s!(g-b;d;0CM~^)HM07!JZeZTnc(~c&WoJlsCT0<(<2W5Bse2Y;W9)zO>cOu*-T%X9?nv>^|J*o3AAIra<|e+G+V3sP<q=KC1VzW){oc*V5%x$s0Wd$@DszgpnmX(-nBWDYcG~m)0qyMfr0{{?R%*&~XQ|Iu(F%h>5#H7bK6R#hc-e$FltCgce&OTSZ@u-Wnlxd1TqIk!Jz#~hJ!r<DTlx`*KWlQRZHG>CS10`*1~F$Zl|i?o1%E15?yLOe<3AsK^>Nt6vfhQE1yYYto7Jveb*J4J3{|Mzbem=aU+L9tw<kTCDtSN{s!v|nT-_LT3+xg)*u?V8fsWsvu#hr}ttTu#iedlt|BTLWURlEBt1fKdi}!DS`r7`DcW-|ByKnyKL)--Mv8I0N>A(G9bny8H`~UKLcbxV1`}<$Ned}lczW>H2`|o^s>rLGM-~FF|@`KA8Snl2Y;_HKVe&cxIl|MEw@BjBN_J8^zUXCza(f*e&@4xpa_s|d0<qa%?4nBVS)*s)R`I!$ayfXde!$YSx_>;k287E>)*t9fH?HJFIY=`^cfidc^x9Zw)g?Ex@cJLj)?D)etwYjame7fDG&UW1X8*d%F_R(LyzH#e|_x5jmUXrRL^v%ydIQWm>G(B$;`)*&EbWYM-;nBz^|5{lbM=QD=)RpO-Kgg96kOl7&@4tO+Vckq7@=Divt>VqrxTtP4*(J)}){(Wh(bqCNP1)Bw?PZlW**7u#&bQ$xB)NG0q$n~mI~Baf$l;3wQtU3*%!_92?B!GE9)I-NC*0m$nKnbN1y2{8a4pmdwLG$|xojD#)_Pap<_=zf-L-J~$7fEjpE<2pPCvPF`ohZTOABpC?n+NNc}>%{4kau$q;O{#^?OxHmio<5$8VOfL{M{+prj61P7v#S(X4O2Xxv01ch_2h3bgmOe`C~7!y<K<-epGn7nUCYOP{&!$g$`0pgaGVj{o7h+yT+c7ovZRu6sg{9~Az=>FK@kbz6lc;-_1P?L}vz>lGjDxseJA^t^s{O47r3%AxFzd-!~NbEi(IoJJQOoeG{mX8%iV_^p$mC;lf%00;e1ibix*Zs@fhzjO)Xj@TwUm^ilwd%f@=7Wro5A>aG)kMYl$3m49uI~P6q<am93Jee$Cx^!vp!3WB>K#&K0b04aR6g`;XVR_3vRM#ZR@-%z!0e>;B?llDL-S^b2?t!Rv@ilGh!WX(kE=f48Au39>)<AunedmGle9Y+``bp3<isLbjeaQWh0_B6|AIr01z|wXo7j8erI|j^(tDnE`g8x<n0J;n1<<XBYLx44R{J6c=IP26Wu#a50c)kvd_PN@=de^x-UOXQK4%l-s!zyes7h!E>d44^*^eA(^+R){Z`C(tX$*KpORpVsUJ3KtDF+nP*tl;d)W3MeWFc2kFR{df9*)+11v)`+Sv$Ny{FNM%qb?I?aZ2)G~9NpEBR_%H!oa`UlHqHe^*1_mqq(oFuMszICFhp7kH4K1Tfzj6sDr@N0`Wd)W)rzSiPsNQX#IbR&O}G_}eC&bT27{ZmD8Stsr1KaW`X@#Qv86uiohxJ8Y;AAqUA0=QuC4A|8#<w_T3x$DRF{jLhZQ_th=(_8{`c!gw_+K2!_aLjf^oxj>cE-*@u1wM8Exv$(tISe&h@P0o6<KZl;2)E>z<sge;flkZsxQ3Xma7@{B~DdH=4q^>s8>W<z_bzuT=BG1ChOa%5muRhE4ZsU0Gt!^*8^(shcmY7`eT6ZN<AQO{VuMr_Afdtt+_ac_-X+Z*VK`c)}xMOz(KYE2mB}<rvs(6wR8Wj)^Jiw)8^~&T=bh5acO*wd(+K^Xp$@5nZ~b-+K4P{*70$<iGjZYy1EDrrZ8``O70gaWv3TUMDZLcu{_aSPBuM+Bu_xB;-t;3i_*=HKv^vhB~+8w`+oPgQ}P*uj9InZbP#EJ@ihsI%<CR-U<5Rw#nP8DgD-<J<W7{y|12t%6%Tc;4(&sk<0bYSKGAJiC>isY<-2Ru<l25b@672IoGx_SgVIk*zg^iM5m_+Ivrv-w8ZQ;2^!>jpQL*$)vew27H4a(#KddRrl$7lN`x!7i`ro-Z_6x}SXRBwwy<9n!$c27bRZs~^c-{U_3}k0ww>g$JDbsUS8~@WzOy&bisqx}&25S2t#@9&`Q<Mh<dB!sql;u9f`<#&tmYj{z@~-guT&_G$l!&`n>mb`c845%IQRBB*A4J(v9BXYAGfl`!tq!DurFPBm#0%bpFQ`SzxVu7`Rw^2ze8VU1CSYcckH@`#^xS-nBk~b;Zx;h_v6aJa9s9a+>{mVT$K(1d3|el$22Ck+TGsv`FlI{YvTeg+~ua*0bH*D>tH2cf!Q({7CyJqAJwm{G+%{u@xSJ{$UIYh9lf-=wuV9#gaE_pjRG2t3Ug(3!<D({MvW|o#!^M0f%=1YYMKo8dUmraW|%HJy1zRlJuA&oq&3k%^_}lb*ajOX+#e7AwS;YatHB!q<Y2Wr>Ov3RS#Zap822X@<FVwIu6`YiA{>J*93|}Io`-x4#C3F|Ku$dV!~HuITzxVT9$~ARl-_|UFf+n&^4x;Yn(afFpZ6h*e{;#TYka+I-rf)Vy^Cdf?CVt}doAK?FMgx}v<_oVOdWGF={R{gW?6Br6Zg)y%e%&{fA89LN3LDsUbuFN?_In8f4g?g_OiO~_^w^|-DTgd8P(oX_pa_SNJobr4GDQ3#PK1B#D0!_`tcw2vkrwCsScv3-L=|BP1H#EX@xXV1M=1=$Nug)(?qG%n!B@ITy1_21!c9zKW-tiv$+**+BVp&I*vGh-o1xea#gwM<=P~=YPQW_oYlJ7kuHCTPuFq^rliYD<z|pF0n77y#|y|B*wH@9j`fV~>Dfn3wMz6p`c1@?SgHqd9PNMl!T!6i?*IJV25EP0l+(0gEN#6>b7a?Dn75g4>Yx1$!KQUix3(88qxEX5Lv*GYYcxF-1m|ELf~*1JX&>~WF0tB-97bkI9fxoG%l8g`_3^EDZ?rjehBw`P*)<=Z?XvLbh$k0wPABFE!BpWK#jdr}L3uS$@k;%;!cMLbkz1-S4v@XB;0df;Dg^VP;>#nzl{Ty`9Ioa1_KZ<I1Vp*-qWIETG8KDt<?2~#MDKP5v^><n`vYcMT3l*cHi)U~cOOFYWqROy*wIz%&#g`r^mrdk(6Qo79$1`LO^B)8E~n-MGu;<GTP6-W^vwpS*+X9%?MO8PCAqzRY|}4ZHvHaG`CU{=Fm%mk<e<M_G~EO<Gd4*;XVsQAa>6+3aqeORl({S?Zg58J;4B`I9FC74dFH7j5+noyyT8H$^C#-jrgwoW&({~nvp`y=fWJD1Q(sKVzZ)&J6cjcn^iC<4<SvPSh~Q(j=4{Mf4yu(F(YL3O7l%S6gW@W5l`2^4zUZ49Z|uMK#=%D)mDnt^;(w-?48wYT?+|b-a8nn=ZEg~{PR!~v+s5tw)T?sWXH)My9JwTi4;GlFWw`w1Q`K!}zpPFiKU_ZbeDAP$rP&SWrVv}TUJq0yEJhDa9|L(9$?AnEyvt6};p9{OJ*vUFwt9$=N2FZ599=5?I@}cRhUAz<w1o2M23DFKkJ1UtZ+Qf}A2kG6Ei#@7+D_h17hL#fJGC71^AnKI-7W2pVva7Mfk<_9e=99@o2I2`7(BRse&jI-_i6&ifbRn~z7N>Aw*eb_OQ&Ga#fdmIzJBWB<By!X@c8n&25VAdRmN<ZGOLA+wc<7xEU_vkR&q&s9Oss3q72VyTF_ibo=_oelBq0H9Q<B-A~p$>N|raQAd2%i&#a<~@?6@&a+T$|utgs8IF*J|!v)WaoN*<2p5!?#(kzLKEJ@mO&5JvmYj?SH|NpY+B2HuUO%znABDPdm#d2$vN>f9nOjDI9t&Ol8Rr6RRSym8GFQJ)cF^va{uCZ^*Dx`^1{3DZ0(}d)OLhA}8G)XO|sUcPri6DhFxi$r1Nml3#jV`3deL=I~Dz?oPn0?#Vs&1r8raCqRgRX_hbH)mq$yBMF$-)?=6P;=+B`bKA@=T=G7K~+rvX~Uc4%ac+GO8>;PkF&|v``Tt3o1Ytlc&0n1v5-(NpoccQ7RLODrk~QW~5~pq?8z`jqI0S1MQU<e!dD&hl8+!nS|wuEE1)`cT{CD#UC)E!OvKLdmZIiMhw~^3n?{v#VKp6SeILOje%U*4dfFcR-4RZgvO=-U*wV|HYPd-k*J0UWrd=t1w9Ih_GkE`RaV3j?8VbKZ;LJ62Jc#{tKusZ?2==o1POVfER+Bk&t#S`QV0t%PfRYQiVf$5p#=wv2&R;_3B&;H;Up8qjKaIDuT@$UAQ8StSFBOFO|3!E8Okj*C7h95#HmgT9%pGv2t%u4Wk_7)MM~OA>Rpqv+%$#OGFlU!a-rkGaH+s^(tv4HkznajXs(k)QwFBunrN2gcrUl8H?;vfR*ZPpC|{Rf{aqFhO_OF?P^$!^al%;^o17@hEMx`TVvG@4isl#8X!HtnF}XnTnrDjXbY}5A#Y1C7s3n#p2~mt`%QO=V5<^mz>RbwoKPjV`O$BqD39>^Ea*R2*hJ@8rOYXZ!r@Dp~ORUs9g<&!;gp4J%+Lb|Mf(c6t47!#@Dq@K`xr9I&!JzFxSC*vQSfSFG&8?zKFTzQj+B8og0EDQ*GD(bq4we=_EyN{_1&eJVGW6B(g6K@8B9F5y6Pl6vRkg|#vk6mRbt1Tc!X#ScX<o!C#t<nk62?+3xQ&%&38#*+X~vAjC~?ieIH;%0YWl$6?V4Tl2rL@&_jNsGs@O_Qvdrd~+d%T%KoDpFMj!<z=py8v(=1P|09RQ7k=77<jGE0r;)J8RprT&7#1k0zs$h3Wxv~|sIa4TE#5TtdX)yZeH3nZO%XNx|DUxB>In6OMD4xZboVn8*JR^llL{PLlB`Uq5T4cDk9!?l2tA(P3<R~Z+DUBJX#f)W`OjV-L2*FiilY+%eq(ZTjaHBJBtq`WI=O5f4tVlNsGwOoc!4%1Jk>;FGfxi`t6PcS##i&o2LMKI@fNVCVf?(c6*GN{d#H1Nb^0uzYsyp6aDXZ*MkqezFm;n?*A6tT6!HjTfL&XVovoK>Y0g@u4VA>+jFi$hE9oisB>Z>S^8ScVRR|R*^FDj8^ptYeUhOq+!!fC~v!5+#eCyBsV#|0EuoF#>UMF3SmAodh9Y$DM=7PqAj9IRATPzwde1+1%_3upzSQZP5oF}p!%VFtrMlhC+^m>5ivKu1!T7otd+A(?Hf*lD5W+Fa6QCBm_dp)vGoDxqLmmSJk+(!kKmVM#kXA*Q&XA}b`u$cZ;72-}-xnC(!BKkVPG%u;30#tc&}Wg<x=htWnsRWy^JeCP@e+k;4a4_QHh3dDy>1j{lEcI;+|xtZMEQou@-rE-;H@`t(;&<jMTLW1M4Mt~;OC_9Z|7aK+6f(vTEnu?*-m@eX`iVn4NSh6}LG*cKWr;&MPAU!;V2s6c50_GMHmZ*jH1}90F2QWi3H+3tTr!zCOo&X=t$ndIQ6l-Wg$rTKIL!A{4{gcF;=P*LRd>PD$Tv<@8pd{zHl^IVHhHDP~oQWKa@+|Bs*!wia=a@!Jky2=wGzKA*%%s9_!6GqSgXkG#)?#+y45A4$p3a0CwJ|ykqmvj|PdNi^6#;`u&MZ`zf$*Rb4x0iDkuoAvrYL55BJd<rHkpM=>G+Fd#lR?$xg;dk3}%PSbFeT4V?)hmm|mqJyohybQdDbUa#?7^1k^qpp2%YAb7KcKq*z!M9I~k)<qZ5Ti-JIE3YBS0^jLUtZAmPlsxU8Qv|u8R=V5n%!=Ti$j)mrM)>4us&?>o+C?6JtO>!cgg_X$E<ltw5d4wihODKLRG@k`?kmSy0r6P&JV~iv@jCX}~BD8t}ODZ)?y9qTdFrmPNQUw;zMQpHCiDBQWY*xS|=2?sfgY81H+@!D|pc_m=powBAY}haqY5=P$yh4b;)aN)lhpL2HDZIHc9GF6#6l((U!&0mOd9fh02Bjs&Vo3>UgcXRj8&&|cfK|bSTY9O?2ra?YgYJ}tD+KFf^j^RoO*OR!ObQi;MS;fjg=sOvybSG-3WGtzvQIhj6j=sIEoM43WhQ5_hA1$I8w??dSqnqNvHC7Ra2T)|#|i{XZHz&YLtkUQ2G>DQ6om?yf838%3xNctCU(jUOGhXMn}WQUcVn)hrm)K3nUrp=gLNmk+Ok}uC#aUDdT@VFgIg)Xio>!Pssi>WY)}&$>MaLRXoUfE>ktbejv0uGmS(WMgqz=Vd}?p+{{XCod}0"

SOURCE_PAYLOAD_SHA256 = "0238a5e39c68d0625decc976abd7b27e15216293179e1132d5fa2ee5c1f814ee"

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


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def records_from_trace(path):
    rows, counters = [], collections.Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            t = int(r["time"])
            if not 0 <= t <= 300:
                raise ValueError("此封存合同只接受0..300分钟生成的请求，不静默删除越界请求")
            index = counters[t]
            counters[t] += 1
            rows.append(dict(id=f"{t}_{index}", time=t, origin=[int(r["origin_x"]), int(r["origin_y"])],
                             destination=[int(r["dest_x"]), int(r["dest_y"])]))
    if not rows or len(rows) > 3000:
        raise ValueError("请求数为空或超过本工具3000人有界范围")
    return sorted(rows, key=lambda r: (r["time"], int(r["id"].split("_")[-1])))


def build_scenario(root, env, k, fleet, horizon, trace, seed):
    target = materialize_source(root)
    # 每个case在独立进程中加载，避免K/机队级别的全局补丁串扰。
    activate_source(root, k)
    if env == "E1":
        install_closed_loop(k, fleet)
    from at_obj.scenario import Scenario
    if target not in Path(inspect.getfile(Scenario)).resolve().parents and env == "E0":
        raise RuntimeError("未导入隔离作者源码")
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    s = Scenario(max_time=horizon, person_spawn_file=str(trace), enable_logger=False)
    for vid in CANDIDATES[k][2:]:
        s.vertiports.vertiport_evtol_capacity[str(vid)] = s.vertiports.vertiport_evtol_capacity["1"]
    s.reset(seed=seed)
    return s


def replay(root, case, assignments, trace, horizon, deadline, detail=False, rule=None):
    """按作者step顺序每分钟最多决定waiting队首；不提前把未来乘客塞入场景。"""
    s = build_scenario(root, case["environment"], case["k"], case["fleet"], horizon, trace, case["seed"])
    candidates = CANDIDATES[case["k"]]
    counts, routing, timeline = [0]*case["k"], [], []
    area = 0.
    while s.time <= horizon:
        if time.monotonic() >= deadline:
            raise TimeoutError("到达case预算；当前未完成回放不作可行解")
        action = None
        if s.waiting_decisions:
            pid = s.waiting_decisions[0]
            if rule:
                p = s.persons.persons[pid]
                score = []
                for i, vid in enumerate(candidates):
                    v = s.vertiports.vertiport_list[str(vid)]
                    access = s.vehicles.estimate_travel_time(p.origin_position, v.vertiport_position)
                    q = len(v.person_list)
                    ready = sum(e.state.name == "IDLE" for e in s.vertiports.evtols_at_vertiport[str(vid)])
                    score.append(access if rule == "NEAREST_ONLINE" else access+max(0, q-4*ready)*2.)
                chosen = min(range(len(score)), key=lambda j: (score[j], j))
            else:
                chosen = int(assignments[pid])
            counts[chosen] += 1
            action = {pid: dict(mode="UAM", from_vertiport=candidates[chosen], to_vertiport=2)}
            routing.append(dict(time=s.time, passenger=pid, airport=candidates[chosen]))
        finished = set(s.finished_ids)
        area += sum(pid not in finished for pid in s.persons.persons)
        _, _, terminated, truncated, _ = s.step(action)
        if detail:
            row = dict(time=s.time)
            for vid in candidates:
                local = s.vertiports.evtols_at_vertiport[str(vid)]
                row[f"V{vid}_queue"] = len(s.vertiports.vertiport_list[str(vid)].person_list)
                row[f"V{vid}_ready"] = sum(e.state.name == "IDLE" for e in local)
                row[f"V{vid}_charging"] = sum(e.state.name == "CHARGING" for e in local)
            timeline.append(row)
        if terminated or truncated:
            break
    finished = set(s.finished_ids) & set(s.persons.persons)
    people = []
    for pid in sorted(finished):
        p = s.persons.persons[pid]
        rec = [r for r in s.person_travel_records.get(pid, []) if r.get("end_time") is not None]
        if rec:
            people.append(dict(id=pid, travel=rec[-1]["end_time"]-p.spawn_time,
                ground=p.time_stats.get("to_vertiport"), wait=p.time_stats.get("wait_uam"), flight=p.time_stats.get("fly")))
    full = len(people) == len(s.persons.persons) and bool(people)
    co = statistics.mean(p["travel"] for p in people) if people else None
    result = dict(full=full, ATT=co if full else None, ATT_completed_only=co,
        generated=len(s.persons.persons), finished=len(finished), completion=len(finished)/max(1,len(s.persons.persons)),
        restricted_person_time_per_generated=area/max(1,len(s.persons.persons)),
        simulation_minutes=s.time, action_counts=counts, action_shares=[n/max(1,sum(counts)) for n in counts],
        final_station_queues={str(v):len(s.vertiports.vertiport_list[str(v)].person_list) for v in candidates},
        mode=rule or "OFFLINE_ORACLE", strict_environment=case["environment"], fleet_size=case["fleet"],
        eval_seed=case["seed"], control="LQ", step=0, task=case,
        aircraft_ever_created=len(s.evtols.evtols))
    if detail:
        result.update(passengers=people, routing=routing, resource_timeline=timeline,
                      lifecycle=getattr(s,"strict_stats",{}), aircraft_events=getattr(s,"strict_events",[]))
    return result, routing


def rank_result(result):
    # 不奖励丢掉难乘客：先最大化完成数，再比较全体ATT/共同截断受限人时。
    return (-result["finished"], result["ATT"] if result["full"] else result["restricted_person_time_per_generated"])


def cost_table(root, case, requests, trace, horizon):
    s = build_scenario(root,case["environment"],case["k"],case["fleet"],horizon,trace,case["seed"])
    points = s.vertiports.vertiport_list
    sample = next(iter(s._all_evtols.values()))
    costs = []
    for p in requests:
        row = []
        for v in CANDIDATES[case["k"]]:
            a = s.vehicles.estimate_travel_time(p["origin"], points[str(v)].vertiport_position)
            d = s.vertiports._calculate_distance(points[str(v)].vertiport_position, points["2"].vertiport_position)
            row.append(a+d/sample.spec.max_speed*60.)
        costs.append(row)
    return costs


def milp_proposal(costs, pressure, timeout, k):
    """全未来二元分配＋满4人站点计数＋凸分流压力代理；不是完整飞机调度MILP。"""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    n = len(costs)
    if n % 4:
        return None, dict(status="SKIPPED_NOT_DIVISIBLE_BY_FOUR")
    # x[p,v]为分配，z[v,j]为第j个四人批次，递增边际代理成本近似拥挤。
    batches = n//4
    variables = n*k+k*batches
    c = np.zeros(variables)
    c[:n*k] = np.asarray(costs).reshape(-1)
    for v in range(k):
        for b in range(batches):
            c[n*k+v*batches+b] = pressure[v]*(2*b+1)
    a = lil_matrix((n+k,variables),dtype=float)
    for i in range(n):
        a[i,i*k:(i+1)*k] = 1
    for v in range(k):
        a[n+v,v:n*k:k] = 1
        a[n+v,n*k+v*batches:n*k+(v+1)*batches] = -4
    rhs = np.r_[np.ones(n),np.zeros(k)]
    result = milp(c,integrality=np.ones(variables),bounds=Bounds(np.zeros(variables),np.ones(variables)),
        constraints=LinearConstraint(a.tocsr(),rhs,rhs), options={"time_limit":max(.1,timeout),"mip_rel_gap":.02,"threads":1})
    meta = dict(status=int(result.status), message=str(result.message), surrogate_objective=None if result.fun is None else float(result.fun),
                surrogate_gap=None if getattr(result,"mip_gap",None) is None else float(result.mip_gap))
    if result.x is None:
        return None, meta
    x = np.asarray(result.x[:n*k]).reshape(n,k)
    if np.max(np.abs(x-np.round(x)))>1e-5 or not np.allclose(x.sum(1),1):
        return None, dict(meta, invalid_integrality=True)
    assignment = np.argmax(x,axis=1).tolist()
    if any(assignment.count(v)%4 for v in range(k)):
        return None, dict(meta, invalid_batch_count=True)
    return assignment, meta


def run_case(cli, case):
    root, directory = Path(cli.output_root).resolve(), Path(cli.output_root).resolve()/"cases"/case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    os.environ["UAM_STRICT_LOG_EVENTS"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    materialize_source(root)
    print("[就绪] 隔离源码已校验，开始导入CPU求解器", flush=True)
    # 首次导入只占初始化开销；搜索计时从依赖就绪后开始，父进程另设硬上限。
    from scipy.optimize import milp
    trace = Path(cli.trace).resolve() if cli.trace else root/"_author_source"/source_snapshot()["commit"]/"train_data/passengers_300.csv"
    requests = records_from_trace(trace)
    deadline = time.monotonic()+cli.seconds_per_case
    rng = random.Random(cli.search_seed)
    costs = cost_table(root,case,requests,trace,cli.horizon)
    best_path = directory/"INCUMBENT.json"
    old = read_json(best_path)
    assignment = old.get("assignment") if old and cli.resume else None
    best = old.get("metrics") if old and cli.resume else None
    evaluations, history = 0, read_json(directory/"search_history.json", []) if cli.resume else []
    baselines = []
    for rule in ("NEAREST_ONLINE", "ACCESS_QUEUE_ONLINE"):
        print("[在线参照]", rule, flush=True)
        result, route = replay(root,case,None,trace,cli.horizon,deadline,True,rule)
        baselines.append(result)
        proposal = {r["passenger"]:CANDIDATES[case["k"]].index(r["airport"]) for r in route}
        if best is None or rank_result(result)<rank_result(best):
            best, assignment = result, proposal
    save_json(directory/"online_reference.json",dict(task=case,episodes=baselines))
    def assess(a, label, solver=None):
        nonlocal best,assignment,evaluations
        result,_ = replay(root,case,a,trace,cli.horizon,deadline)
        evaluations += 1
        history.append(dict(iteration=len(history), proposal=label, metrics=result, solver=solver,
                            elapsed=cli.seconds_per_case-max(0,deadline-time.monotonic())))
        if best is None or rank_result(result)<rank_result(best):
            best, assignment = result, a
            save_json(best_path,dict(task=case,assignment=assignment,metrics=best,trace_sha256=digest(trace),
                                    source_sha256=SOURCE_PAYLOAD_SHA256))
        save_json(directory/"search_history.json",history)
    try:
        # 不根据代理最优值直接宣布真实ATT最优，所有提案都重新过作者状态机。
        for scale in (0., .02, .1, .5, 2., 8.):
            if evaluations >= cli.max_evaluations or time.monotonic()+2 >= deadline:
                break
            weights = [3.] + [1.]*(case["k"]-1)
            pressure = [scale/max(w,.1) for w in weights]
            print("[MILP提案]", scale, flush=True)
            proposed, solver = milp_proposal(costs,pressure,min(cli.milp_seconds,max(.1,deadline-time.monotonic()-2)),case["k"])
            if proposed is not None:
                assess({p["id"]:v for p,v in zip(requests,proposed)},"MILP_SURROGATE",solver)
        while evaluations < cli.max_evaluations and time.monotonic()+1 < deadline:
            a = dict(assignment)
            ids = list(a)
            # 交换单人或整批重分配保留总人数，不允许未来信息进入在线baseline。
            if rng.random()<.65:
                i,j = rng.sample(ids,2)
                a[i],a[j] = a[j],a[i]
            else:
                origin = rng.randrange(case["k"])
                group = [i for i in ids if a[i]==origin]
                if len(group)<4:
                    continue
                dest = rng.choice([v for v in range(case["k"]) if v!=origin])
                for i in rng.sample(group,4):
                    a[i] = dest
            assess(a,"BATCH_PRESERVING_LOCAL_SEARCH")
    except TimeoutError:
        pass
    if assignment is None:
        raise RuntimeError("预算内未得到任何完整回放的候选")
    # 最终验证独立于搜索计时，最多另给单次回放30秒，避免预算耗尽丢失已得解。
    final,_ = replay(root,case,assignment,trace,cli.horizon,time.monotonic()+30,True)
    save_json(best_path,dict(task=case,assignment=assignment,metrics=final,trace_sha256=digest(trace),source_sha256=SOURCE_PAYLOAD_SHA256))
    result = dict(task=case,episodes=[final],evaluations=evaluations,
        full_future_information=True,feasible_in_same_simulator=final["full"],
        global_optimality_certified=False,optimality_gap=None,
        zero_wait_access_flight_proxy=statistics.mean(min(c) for c in costs),
        proxy_is_certified_lower_bound=False,
        caveat="LQ控制下未来已知的可行候选；不保证接近全局最优；代理MILP gap不可作真实gap；满载尾批未修复")
    save_json(directory/"RESULT.json",result)
    print(f"[完成] {case['id']} ATT={final['ATT']} CO={final['ATT_completed_only']} completion={final['completion']:.3%}",flush=True)


def pack(root):
    root = Path(root).resolve()
    target = root.with_name(root.name+"_DATAONLY.tar.gz")
    with tarfile.open(target,"w:gz") as arc:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".csv", ".txt", ".md") and "_author_source" not in p.parts:
                arc.add(p,arcname=root.name+"/"+p.relative_to(root).as_posix())
    print("[数据包]",target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",default="serial_runs/uam_future_oracle_bounded_v1")
    p.add_argument("--environments",nargs="+",choices=("E0","E1"),default=["E0","E1"])
    p.add_argument("--k",nargs="+",type=int,choices=(2,3,4),default=[2,3,4])
    p.add_argument("--seeds",nargs="+",type=int,default=[123,124])
    p.add_argument("--fleet",type=int,default=12)
    p.add_argument("--horizon",type=int,default=600)
    p.add_argument("--trace",default=None)
    p.add_argument("--seconds-per-case",type=float,default=120.)
    p.add_argument("--milp-seconds",type=float,default=10.)
    p.add_argument("--max-evaluations",type=int,default=60)
    p.add_argument("--search-seed",type=int,default=20261023)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--phase",choices=("all","pack"),default="all")
    p.add_argument("--worker",default=None,help=argparse.SUPPRESS)
    cli = p.parse_args()
    if min(cli.seconds_per_case,cli.milp_seconds,cli.max_evaluations,cli.fleet,cli.horizon)<=0:
        p.error("预算/规模必须为正")
    root=Path(cli.output_root).resolve()
    cli.output_root=str(root)
    if cli.trace:
        cli.trace=str(Path(cli.trace).resolve())
    if cli.phase=="pack":
        pack(root)
        return
    if cli.worker:
        run_case(cli,json.loads(cli.worker))
        return
    root.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(cli).items() if k not in ("resume","worker","phase")}
    old=read_json(root/"CONFIG.json")
    if old is not None and old!=config:
        raise RuntimeError("续算配置与旧目录不一致，请新建输出目录")
    save_json(root/"CONFIG.json",config)
    materialize_source(root)
    tasks=[dict(id=f"{e}__K{k}__LQ_ORACLE__s{s}",environment=e,k=k,fleet=cli.fleet if e=="E1" else None,seed=s) for e in cli.environments for k in cli.k for s in cli.seeds]
    done=failed=0
    for case in tasks:
        directory=root/"cases"/case["id"]
        if cli.resume and (directory/"RESULT.json").is_file():
            done+=1
            continue
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--output-root",str(root),"--worker",json.dumps(case),
            "--seconds-per-case",str(cli.seconds_per_case),"--milp-seconds",str(cli.milp_seconds),"--max-evaluations",str(cli.max_evaluations),
            "--horizon",str(cli.horizon),"--search-seed",str(cli.search_seed)]
        if cli.trace:
            cmd += ["--trace",cli.trace]
        if cli.resume:
            cmd += ["--resume"]
        directory.mkdir(parents=True,exist_ok=True)
        print("[开始]",case["id"],flush=True)
        with (directory/"worker.log").open("a",encoding="utf-8") as log:
            try:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=cli.seconds_per_case+180,env=dict(os.environ,PYTHONUTF8="1"))
                if result.returncode or not (directory/"RESULT.json").is_file():
                    failed+=1
                    save_json(directory/"FAILED.json",dict(returncode=result.returncode))
                else:
                    done+=1
            except subprocess.TimeoutExpired:
                failed+=1
                save_json(directory/"FAILED.json",dict(error="本case超过硬预算，已停止本工具启动的子进程"))
        save_json(root/"progress.json",dict(total=len(tasks),completed=done,failed=failed))
    pack(root)
    print(f"[结束] 完成{done}/{len(tasks)}，失败{failed}；未训练模型、未使用GPU")


if __name__=="__main__":
    main()

