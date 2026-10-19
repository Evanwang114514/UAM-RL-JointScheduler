#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""K2/K3/K4 × SPF/STTF/QTTI：只做作者旧基线的接口适配。

原作者的三个文件、环境函数和评分公式不修改；只抽取原函数执行，
绕开已经失配的旧启动入口。K3/K4新增候选复制V1的QTTI参数。
本文件内嵌与60M实验相同的作者环境和原需求，不依赖其他矩阵脚本。
默认使用123、124两个回放种子，共9个条件、18次回放，不训练神经网络。
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
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
from types import SimpleNamespace
import zlib

# 只限制本程序自己的线程，不修改共享机器配置，也不初始化GPU。
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SOURCE_PAYLOAD = "c-ri}dv_bhkvRHQCgjNykU<Cl35vuqa#oRKulDNolVs=YA$hPMFeDLy0D=c4Myq>h%X-;*Igw>cmLGB~TXC#M;@Fl*J-)&H5V;(X{L8<OaI2~x)7{fE0A(kez31nIBB!Ujr@Okks=BJW>f}JvZPZ(pcD=bYkQ<n9w;Qe8_U&_(_Izh1S*kB=KiVwL&Q?mpJFAWP;`Y~ezPf+cKw<#iEL7UiVtiyIy~8bL(zEH2Y-X&K&bkxXbapgboSm4NP0wUgshNpldIZ|cR;q4mAa`=$@SdH!_wPwAl%f5gLtW;a*P1iMnzOUg9DcP~@6^i9{`yR%T4^siyXy63xl${(EA`qG{O{k~w%e@Fxiz=mX*sXE&32_xZ?>HyZq+SeCFfwHU0JC7mRdNo^`>*MHoV(iC_=CB0q1D(L<MR$7b_*Vm2-AFyIr^89Cj-;s9SOu+*;e&=N6l_N^Q>B*=RKD#nL=}d2rDA-=F;cqH}N&8@tD`jE3((9RQ=-a$pXXTHCEwD|6UQ*)3ICFpA-Y;xX)Pwpm<okJp>Woc4UN?NnXpsW=Di7CKcJbG=-wI`x?rbWekar*g4fba3QNZG2TRzINTIv+26dqJ%|D!lF`h;PYiR`yzn1jk9m@*(aT&^R6<gV!7OOTR0I|;YQUxk#NdOwc<jhl%PS(Yp`DsOXSuTE6sWhN7iy?mYiSL0SMJ4=%m)Fl-(u{v{8h9YjbYXVS{NUoH@4AZH8i4yIjFR!z*Pf?PA+)vCkKwD}c$OJ6|bPU8huR6lnw%m<cY%%jL?VQv}%8@>2sP7>(PU8u-Oj&2e6?EX+BrW(mr19PQTjPH}FbRFG3oHfnQI1J3bExjhe+c4U7}EnnUaUH(EY{>^3Q|8kd{SKMN|1AxNvVQ1U6{kVF=N9r9|9_I)S7|^K!6<`RoklVK9oV_rdxh77=Ik-CjgjU0K%jUd0eBSQsM-T2Zr;8sx>MlTK00|$Fv*!drvm`#b2iCB@)N=M6Il3P)1$v-bZPBdkHr!g-t(BllL=}B~3G{#~zv(~?`OQOz4i3*0TW;AoRIgS_OWJnBufB%(bEsKAQCWabSSrz;Y}RKwtu{>$zB2>MDR7#9XK$_4BxHw94!g&T%`$A%7OX9P<7lyT49E$9t=Me2usP^GY(wa10amd+@4)xm;a1yi3?HxfRa*1K2JJ)`!i?L7=_3ABXfL^lh839WQc|w?m!CSX$nCMSS(>l3VP^plv4*~R2-{n*1M0J|y_>G<EEInYpMh=NfwAVWMLG#HV;HihR;rirv%jp?OUG!rummQhuY!dh!0xI$Rp<mL!7Rr`7;*+XH!uOd@AuD&S|RH{+pI6})o$Z9b7?qbz5}8E5$vC@=K;5klg4geg@H1>W|-G<Vm=zfZi6Kw$`E%tEu1eP5W&~qQ?Ipv2(kh!x$@2rkFNY}`O@ERya|6eN5%RaY2#<+Ds1B&ov*ZT#eWU!fxj0k2s;LNxuh-GB;mL~#`zG+mDyRhiQr>6-qdz0ki}oR)%tO#h<gRr7&<OBD-AX|#MllF7?uIL@9;SzLeUreZ&Mpu_HEmC0msCauyD;zEw^pkRP8_i^FJF)?Rn&C?6=dbCL2riVyXrlv)C@wXMVlCP=Nm0g#x?-j_H@Xi|u-~$6NZBy`tqs^eI(~MB*z8v_i5d;mlUtYI&+gHQP(HPk5z+4TgXGCHqE__0Gw0fq{}Yc%2mzs@odTzXhOrCDDS2+z}{Xxbd%YTvIjr#VX{GnmbOVoWr$c57<jiZ;!=;j`#N6?(!S!Z+{2>@~SwVoHjZquRSLB?%ubDRW5a!h$97w%?13d0L3JHe;OJesMlOpzg=w3x&2iaBi#7*t?tTIb%ZlTBuYz#W5?%n&TO>~VnUuRObiz~|4Omia^s@C<f?@du#t8SNdweRi7)u?D`k*%sz9G)cV&0B2#l*ROJrgxk1yl?(7Nw`xB32G+0cG?a(?~V-L;j=Ypc)KZ+-H^lMAdC5NL+%1*EqAcvHy~!T;&+y$hFE7v6<Gu06fMI)(1M^Jh1I=XX?dPqYig#bTw3tRSkljq_ivU%9uw`upxjpRnG<kfBN*|8u(6A9|mCj30S^X7lzce1ytbtyvHc=G%obOtuJIt#E8X`w$0PwOFgU<pNT+Ib$tkrC$U8DioW|%AyPFJX5b%;VmtX@*XH`vDvm)5KG-Cwf*kX$rRfFSOb>GDHK4oZ1Y_n1IcbS&Nl%eG(0C2oPi#w2=|06N6DcCf*Mxvn~H_b+y4`<+46P&u-yIRPG3!M(3O_InrzI@@Gl4i%GjGOaH<*~FC!i`=VV|rCiMOJtDSp~?mh78DHTSgVy#dtmE1<VAThp0-%U8R&H`-<+X%!Es&0*HC$;?&cV5EQ&dVxH1uWIi&lg*TR)<RuZH_hk^Fc(jgW@-xN)woEUTxrPQ$Eab5v7dI0udEl&O!smdTc@M3B?7sCmv{hFH~v;-muU_9%x@A70-tKG5;bjzN!KRW&tFyG!H}(2!tJRTQ&l`>)&=Beic^OsH@W&+=xUJ&YW4l_N{m8u7jeUv-arT`d=S-Z-463u^#BnC>o<A-OG13o`2XklH(V}D!bKokp@-(Sc-)<r>jy7ofPO*1kniAQAmbZVRccq_r3YGrmQ#RNs?4#wX@n<8H@GGzW;!wQnysddJX3dAn8rSSBzG$^-@OX@&kogsMwk>Fp3_FtAlo{(Zd!v#`+EnMFq?+h$e>(q7^Mdi2np@^3Hs-dF}q%^ZVZCmwtG1bMu40bnkukx8+;izuX2)?ylbU9{$<E=6_rM$WhR(pE+T(T;x<<-ttWRy{NY`w>e(}9lsL(BS_GuU<C4zS|LCtm6sMNhP_mG7}fzXQmYFh?bN_YnqH-R%2~h@=X?>Uqt7K!BBwcU$>3*G1A#t!06pPM4eX&KI(=Zqr4=+gw)Bf^$KW`Tyhg5-MbbphVFKA`rF{zOBOHzIt6v&)HvjPP#_fw?GMgezi9KzwFO?iE6i*aL$pMUKOv+H<)ZOS=AK1p}r<>=lY<&5acm8)qM?v{cACw7`fTRc2V>+_`NRDg(wR_<Yz_9?F8{d8p0QN3@10@<zgS<8$HrN1WSa&fa(xR>+TIv+^N3&?#fl06_3(EG1*8Tx2dXf7;s#G$iew4FfMP?Mc!X^x&v~lH&wMWbAUtR+8?VbOwd;M|uoA=bk4Yo^`P_Gq>JA-RaZh8-%Y@GY6?YnD_KiT-~Pu)jn*8jK?Ej1i%cHDq?7VOa%fYg;1D@3c>kL)tIRT*Hg53aIr=Yic4dj_58HA@rV-#E8%|Jmk^OB+|8ZY;kKg^U>av+RdM^6%P5x@F|G^wA4}jq1kfogLgQ^(+{Ggyd6z$wTk_!;NRp`iH%~b$bEdCS#id_SWSs^RxQ-6_79%D1OO6Q;iVQA%jjXuB5<FCt-dp<iBW{Id1kMW~vVF7rl;KIonuzNsxUWVc=lC@!+%W<vZS$KlFWgK|EzPh5+5?H07ir@(lpe59Ay!O(o2?G!I#Uo&t3Odra21?Y=;Bb9>3wmVFm!tR8PfJEbg8Um>NUx-MIq4fM?;8yUs2(J58Ot7vNrGo4Dc>^7O!mMce_#ag+(U{%fTuE5}(eZaL6&d%CWB2rgN|7sf7Eh}fksyE#^7(;VOG$Dz&aOBXQT}Sjra+UnQv~Rt>=kU?JhYlV-S|F;ir*L@ZQQ8cO8u*tCMjUBDv>p)RH5bH-wIZxjl9e5HTMabM2<sVdHjL(@<D{#nRxMH7X_Sj?vJ)`lCpUpOF(!8;9HG@nIF)i$Xk&-D6@u9vaA!GWvR&|vUaiu+Tkce$hCXmr+ove{6nh{$`AB6lE;nhEg-SUfx6z_N6GRWJnP7m!`X^7Bau+A|MzQvBc&|SL8Pp5hYRh#drv_3}19&1zPYq0~GeZ?oP{w2s*gSV>{po$S3hH9^8X=pc5V15kmYOcI1FFG+$?#O1J)PcqdeYRXW(R-7>(@T%p1<l{dawKOclL3!ufbl$flm$m`@vuSN8!k*S_7@9g!p-NvrPI8r~SGrivvt5wC5`=pWY$+u<e>jb&g7J{Ht<T8wZ|BIWJ&kP`QOVu;IKIc#<B?D#cm{O$wc++f-^QrA7n6CjH8yB+xTILaN>JBBLtM%Q-w2_%%g|QPIY${YJL8=AcZ~sWLe8mC-cv{OETRS<Qr{T@r$|jz*x6(`txJtUVfZz^SyHTD?vChmi!vyFf^*S4!;6p;MN4K=L5G>>DQ$NO}CngoS<ZwqnclNw<aU;qOL5{AWf?IP3Brf7kuyu6On#NcIlrbKy+^`eICIJyA)&xg*dE;`L&+<L+rT>&@7#P+CFDlcJAPj`#gvyn7#ju>Zr8H>L*S20ar8BAvX-h0y>Kw-W0~Q5hjbe+QSJm743&bQ{$o^MC4pO`vECOR#bK&f3#UY$1MlauFc2e(lfR=hrt+-|b$xy7uhD@D4O%mV!5-u<@ry;)aW|ky~oFw6i%ggjQ<pkYhSC7zP<Cvt+2YjVQLN3#R1Bz~n>?p^}S;Jgr#Vz$_*Nk}*Udqc%yD&wT?g>(RFiN_i$Qla1%sybnM2t~^-3cGr;z>z#jp<M!Q<H2@TY0}7u<*d*A_Not15__R4<e=TU3zA)wg1v2+^*s!}7mb<qu@L_W+zkSRnO_(MTdrt;+^~ymDD$(Cb5K_&vR>XM=n)Pwo3O{i{``+g*{mg}$O4d>-v9{1pUC<!v%oTR-f@Wc-E;@PjKn?ATrsIR*r=XG{JKd2yfl1$CcyjTacmJm1{kMMet+hw*tUX!=@lwTo6MIU)TPtNcoa90FHgKr#c}h+aq6ITpKKUe?d&lo>T4PTr4EVz|xe~*EUB&CyPPX8Sv19I1zFJ(EDHoj+Ip>6iL(l^Zt>xxXDHfC@{H`Wb(_sg0adK7sq_*O#GzeV7J9B1j<@v_>ublOJ7of<y@@DtFhw$&l@@Xf@{+p^PQa&ET#2HW?4RThAT<Q=N>}PT~G_B7pIoMQvrwZV_%Fe|KbD&l$3aJ*lAJy6mOHJO%q*BS0@=i$6_-a&tWpy$D_B1l0Rf`sSQ^{;r9ZEpBP9-yHZ{hN8n%9aaq}^j+-`;(DU)x{!_dQ2n1lId1)k<f<`FFP+1#VU=49;U=a2^kZ^GK>cIFF_JhclT%#!|f?o$U|O9MDs>Q}+Eurx@B>iUtP(p9|G{puEHmml}K%PRP+v2WUdVFX8lYgtG%6;cVsL6$PV!ZicWdV4kJ<qW&9yD@G+oy4hId^7UzZWF0FI#Qv2>(}3q=s&9KX`tp(1|6B8+U|LTffyJL822KNy`*2$cEEO{vgn2Nj@C)4zTY+bPhiv6dknw0fvkKpF(cercP~q={Q|z?szVcHw62!2Az%BAP$Wv{QZKXYO{f{e~@BamkG9W~3eD<e}ho`-jFIWriN2fO$ME>PHJG1uq1MmK4&UR;Qb=7<OHf!<AgF6rJHk&|;8>`)0D{GJLv0BU;XpBYp63ANI6lZ>&S@>ivVKy?p$P7}l*8ZJ)4;<ZdVCR8d3J73atAFjj{rUE_N0*oc<*&g)u@UMwjql)x6%P#!P1SaG+Vk~h&N&2lyz@V%YP-pWIFrhZ45u^0sc|Qh$&F@n>G7%BzGADrr(9_trDGdZ868fghcg-Ag1K}yH#*^0sW)3WrvY6Smw<L>Ml#uv)M(~kfAbr>b~RO_FLOkIWo|4v#g<cRSS<ki$RCCw$o^s@#txDKGeT*sm4Y@3t;%m<>~tW5b0?nCCxKIprKS@h4Jr#v$z87r_v6XQ>22v$D#3nD+rC7?F*4em>5$6JWOUXi?|bxmfWL~zT)bK_8>*ykG4FkptWvqjR02kv&P}FMi8TC`$xUWQ6Haz?`c$tD3b+S>??w&>{wcd<wF-d3LD&TdW}{Gbaa0nzB#H(qskVZ=0a(p{<iWMIXYY6qSDf@baPh2n=IhOK7aiDK>v!KlBW-%#J9`GUyfc;}UmM5pKY4Ja`_b)<XOA84vyZ*EKkdGA$NTi#^>0=|V1-uQYme6+t!#Y%VEyU4-iOa%v)#Y4@#ULLm7o$bplz*sEp#98*X-Apc2g>sP3KbC(A|{64VTU4GCQ`mn?iO@f?Lqw9M2rD*+x1<1K$gaVlgrul9$==t_<)?JZQbPbHCFnxwT@mQcwEzDvK_^WwKB#%~xt}vZw7Q)Qv9aH^)>>zM9B?Q*T!0fGN`T&I;~Lxup%iJbt&*Rx{{jr&yMqy=6Xo!@L}nbL{PpG>UZ5D6F|L1L(^rG3whU!`<S{;;<ndC52eXczjr+8dN?6M#1iUii#HZC3Yz)ssj3Y6sr+6xIa&L1rDm-zEmoBthMcs+RdjK3atefHoE9VbA_E6V3!`J2Auzb2iP|O#U#5)mNV3B+;$nVbj{)-$j>czj@?5I8&|tfMn{@Lqf$E7X^@W#5V>1-(V^(fB}E6b_V^v|>gA2+AEFQBhkxx}e;DZLcoFt%r?_A-O`=cilC_+D9>K2O)G?=XxKnFKA|rPjnHs3M#{rTWgrNvzH-=^)o^J;-1rUe&+Z~2DRM>PE>Wi*<3~6;{3Zipu=?0xeXha>u&tlmjorcwi3P;{FXD2nz)P2j>fGFz5x~^(W1)MAd#O-eZ<mQW0fguRlyC;h8tIxHf;a|WDl?bTG>7L#?zQ2y#I(SvQ2}d!v&!)&E*rIX~j_omS_^|I8#fA`%uz&0-R;z#%s7>N54Az+4^$c&M?{oHtbt9u#Ub0ebomN`NsZJ{`75t4L@ZIOCeJlpM1ISD;TWqxUXxTgS=kBcw-nmcL9(~31`wnfs?q6<0Ih?pSRgrhy>k5G}EkLHt$L@>Ea4uPF;MLt&3~MJ;pDSY*$K%$^9CDKj-LStgZI?Oxx08DNY3C)<07cvvQ5PmhH*WAq**$-qUFGw>|I7O76$z^K)f>VwDR|?9uJji;G3$BbV`_lc<lm(;ZTH?Ekl0_i<ULzn|M33$&9}T;58>a<<!kalsehE4=lcJuAhGIa*({l}ru?ksYdT*K+Aa)~7bJWu7>|Jnns}G|l_UGNKzIz`3rC*N-6&yiLk>YS@`^BE!#kB3O$`v|R0&Wh;|Z`@`=Y7CATgx(7<3l3)uAw<BX9@PyYiLy#!Y_HQV}%>?gst^0WW5wf4R~!&(|+DM8iyjN}3P;;yt_FUH(#B3hgzNUj#03St<=nO|t&gUg%?6w;-9z2iMMW5o*x++pRMW+okM!hGTJ&iNFr#q}>chkzQ*c<(ryqbpQL4YwtSUd!KvfSGyN3p=}?i+1sCbXYMF$W%Gkiyz}qdnq$kiKiN3_)FEeX!RUeRF)7#k{H^t?XKgKYO-V~pn&5SbNm~=&%N_U>8e+A>g{{SOvREz`fPMPc7-IY!88_jf5MF(aEh~`gYw3K+(gynm1tQQ+>x@)K-<naGLG&5VJYl}XF{0kaI&`+77qR#jNd0SV3ed9!Dj)n_h<A9*CMD=ctCy@*<eAC-8c$6xMfA;xx#F%Bypo6-I{X4mqtXahrl3A3@=l^kE-bJ8?T`rSNlZ~0W&X13fRydDwS%ft?lg;j2oY2w;jn`}1<okWt5T-q09X7$^%KjWv+?`}h^XBU9?0lP2u_M1h1f>AF{U!mWK^^qtX$i8aA)o5b96%Bje8xbxsO<}T9S9PZ1}=HXq@5#%MyB%jzQs9fKPHB#kK+j3YbBWY)&-X5>xKRgP{4_%}+7R3_u9L(*>G(>qBq(3Cw)$(IqNbUSB;AB~hn;g*2j@2>YO0W>sh#^*>1v2R{51b+FGrz_SI}I@a%9fB~^Z1i`U;{jtLs$2v$0m+!#R05kCJe~eBCI8qy+ULogo$JZ+8w{Mvp%$P{A(hOU{z!Z#iG<J`5&uW?yF$jKZ6gRMbVKj6H6lXDx6RPA95u%hnT<g*sKaJ5REZD0k&U&Dzcm5;q%Xhqs*HPr=3uhZ;Uvile^<utWZ@K<B|9+%S9T*)=dlS5a2lo93@QPBsS(ah&oPTlnmc(#%01HKbbl@Pv%<b~Jt%K{+sWOYmyxVk>QAAPKMM$k;m^vb;dRl7GIgwfx4j+8&=pN09gDaRcKw|w~+aRda3f&k%Rh3<d8$mpJX><eM?xB!T7d}!kqgPb7fIMYJpAolk<WhU5t<pj_U0-qK-TqULjb%}TTcuPhti}KQt0o#9VM-j%+En9%Zbn(Zzl^s>zW&lX_vz-ji{7*MH@^R9{fkwFD4)=c7n7-JXP9@anGnlMM#Z#JF+Cj=-Js`f+nfwus$8PK;`nbWnKj)VmDqD3H+TH!*IG<}czB<4youVWW>077z$Rp4mRIKtb5CGrsT5eWu~~aw=&qu~LBF0T3Do7PPf?LQ25NB?sHkOXL{C|)tL&G4C2(kH$mz$9n$C@+bE&bOUJ04e7xGFFR}%SSskQh**AuJtxlXeqYQxhUB94AzP#BTvm#FF+X!XkIyn(iOUTuG|5j5+DUmx@>yZj1am}xiqPN5oLgkK^A(SONdfiqzeL^h#kHgX4p<7&4Ll@KNT_Mv{s&}FgQ>a*&XY9KAqwM9<FU)<0P0L<ZX0bxOpl6b&v5zndU-Gpxc!tG-;V3)>#oNDB>Y@h~C+<tQgiR7*$uM=osl5Oq<W^)`7J`~Jv!s@Z6j#s*_sfAWsFOFZ@dhx+pi$<7NCS<jGS!{s}p|wkzK2}?AYGUsNjEu<J(9IAs*W=@K&_Y_7RmP>>&9W)v3b0V+f?q%8|G5>d!#?OmMPJ$w=-$Gx9Trv8r1d2qmmKYg1gH2X#2-ru3TktL{o0BmBRCi0%LG8gCg!5AK!JkB)<F!2$vO#Vs5z8ifly&z<%c@$+2Qe_IQnAE20Yn|Ws)s`5GHXk4pTvF)^M|Dv2Tx~-Gw!=Xs2+qz{<v;-sBL%WoSActOZNkrsWhS93MF>4p#vTnq<CR(;><+wVp6GowWBm#1?BP=S<2`i-m-dCmgeM2}-T*WZ0R)z_lJ3BIx}wtiaxaot@eW${xnc$!Z+yEnu4cVsjt+-n_TmkKxf^E?XETZH0h|G5YeA|7?C3H7)23HwQ;k#?6$|*<ljzT0V<nQVa;x6!sk1l;ly#9_D4MB@A`M>`uCGc;_na>*v;1zhk%gx_57TPp+tM1JsZ+(=sYomW9HH$Gf`Ny>(&z=3DF${bx@<WoBJw(Sr(`x0W})c*7~zI~c#ZH1C#<S?r+x2Y;5gkzU-#ILY8PwrI(bd{h#QnnDBg7pCCLF>}*GkJIrGLeOAGo2h}5SnX8dWaZS<04fRZe~x;m`_J$)l2u7ht(h2R+yrH%u?BkXvu&QoL5e*;G_9S5_&as4+en<bV<sEy-Y9#t*XOt)XjVhJh-w@_cwN)9&~kIVn`nX3ZeHo$TCrmttWW_?u#5|8v;fz#K#TxA<YOVyN<4@z{6;nV6^lDBn=*@K_}VZ`n>}HbKN7XB`FShu7*%>8NZcZN^s&I!g$<l3jrHk+rtYHg!R*Twpaq>`)hBaXkvH^dhDz3w7MeOXwol9$ALxZS9lVpIY@DbTipU)BXXUuWmHu{XF@G7jTF9>`2_!^*j6bPlTEu1Ir!aJV<_tTP%-CMW6Mw(1^`7->i3$WmCa>1$9|Gf1!<_h05O{&E0}wC~5oYFNM$ER4Z2=dAl`=0`bi#%>QUQ~<T6%1%ntX4qI@bmm4v+ytAV^66#!`0fMibo&30VDH;}4a$BCg{LhfP!1L849UR_H})x^!7-y9>eF`juH7fcyXvQ+$aMHo1W};+|?a#DGoS?3(QHWvl&NOsWSlm&Wt{SMh&!+~!hmrAS{ZiYn0K%)D3-&3gnb5FrQ}JOBY(ps=+ekg3ojRNjg&>8o&T;o5Bxr3yd$!blt$^Z@-yEpg%+$jll^Q(+wM0|1)NQ)<{q%_&X;Jr$kl9l2CGH)6YfmBlLQvA*@2zN!qbkA1Rn|F)vXpgZ}OysH;CPouNAJQ_IOrEl1+Iq&nEYmf2jjd$ze=0~5&NA{M-d!<6AlC%YP+iGeJ*$b*pDX`d7jsKaPDYh!5UG>^*WsZ|}%s5rei}!mEymFA!WTu75PYp~S*|q1u&ck~T77pz^a%9heSN9yAc77gPa9f!3B;Jxnv3VGE6;?B^a7NMbkv0fJ3;)JyXxc+j;}tlmi1$+iV&#3upCnYoIAVAi7DjbkhN6MZ5$D-s9A+nnBis6)-dTJ4dvvP2dU<{I9vsTNx8c>>JaZDdchdF4%|CtMt$cwC^t+2}ExXqrdspA}D=dFP7bKY;8l%DT@d9%1qp}z?5DF;KxRh0xTUw|A)H@5PBZ7ym>ck^e-@*={tH5Q9AU(()A&JBwqb0|okfy=-e1Yctg|!1UYJ$;F4c_--;}b>(9ApTx@%#qfjZ;0{*rhkT9_M}WX7}S;xbELRTw6K2_Go$I*;$S;&gPqcSbKclJ9|kz+OaC^j;gcv{G*KrZ@@bbUIXUcf4q6_3O@<Lcm2K)xkypKT{$G(P)5jV0w#MOuE62j`}iI2%J<I3)2Da~H{-0WtT>K}p%|FKIkvEU-f<A$&wu4)@tNO@<$8%s7OzE9*y0Bp4^MBr{U$0%K6uc*ey)4_4;v3wy(`yg<Uo-wI8y(?TuRUl_1a^mPQih)XWx~9O~YHh0Wk6&JXYLpm@grp5{r$&J4YDYx*NbA-vN=;#?LVw3mfC+^39Fs=eTNz6(ukm4Eo>SzsXZKy#F1pDQkvX`|KsQX6VPm!XVvRChCBRV}vzGn2ckvQ<HmYi*cJp2(xLmDht9BK~WM)+t5o&F1uQb_Kq;u?7^H+-0F~~_3--=bq?>7(n*VwPJvWZt&pLpx`h5%c%{H(=flpCPJ=vkI0GSb5|CA%oy})lb1XsEk2DCK@V!6K){{+H1N8gfDu7(4H5o~zg6H5~R-lBgNMR9dNH-!8PV}GAqF+#a9!ADx?j4K~WAW8%bsET}k`fB>pkoQ2J~tg%+#w=CfEta+_X6R$3hDe;-Fu&j>_#CPWmW&gH&V;4XO?F?UM;4XtpW=Xx8Ij)`7w$`Kk=}u)U2+)%VS~d2X(_<1wzOXiHEJry)#m(2NOE3vQAL-)LDqx^@B_ds>K$WeCkI>Qe5BoSc*y<q>!jdfqU(dBSyuWI#8o>0mSw~WZ+h^0Z2cWGk3i|FN^46_ivN$cIfSP9-E?3KTd*;HMCd?of9&t1b5)}p8tu+aj;?<D~`yDn0Dvh1&dGj)If%6W_i_PzY>)V<?<=rXHf}PCE}C{G}7B?ktFj7uUI8+>NCT=*D->|bcmioah|>Z(0lq`_raC5M_=)Mdh;!I;hw$HED)bQ@t$1CIVV+B-z30$0A*P`4G+9}_~2^?cK;W-s3n*?O0zAZm$g#~@{B6Wf^>p=p25RnrA&|QI(x{RO5i<AXWBVs-TcCpn|b1CXeG_YN*p`ifUHhwpXqAg?A_gR_{o@nt#B}77=iP1&ObL>>^H9nZHLM3j2lvJnMFhn?~8|hQTPCbc~PDNVakF{4AliIOt#=KfCs6VI34Udy!YtdT|4*jBoSYAFZ|xS^l<Iboy~KXHm}{-c<=zv9DlyDaqTIYZnl@*na<o+j`>Rq&dx)7U&tznHv-*qJ`g3Eb3IWJs^CBIEjlmo_VIvTDsbc%zmK^~cC{3=QO((Q1P(D*TPq7|f(0=IQwt8?Bo_7xJd~b%8BWQ4D{n7W)@$d4^rUVG)ugQXxZk_Vb5H&7<Yqgc!t9RTne!lTcUNz-GYmVxnDf9Wif#B|laW!G^eOXgQ6Uxb0PGBs#|Sp$@GE#bODf$)9ivU#M_=?6P;n8z#+q@BdCvu~I^c#}b2aH3FTtAvxbFAGwP+}YlMge~c$ULFkb}W%RA|gj4e(eONB10ZPRjjs3JP`}KFYno2Mn+AbduZKyVoBwextsp3sO?Do1A7mSYFDLCp6o|IgwPYx5%0c8sGenYWv!EEJ$*-aA0=^QQ1AwM!m1nZGiVGbVR9UApi!zCyyN3`Ktrd*fx*4$y20I&nv)}N}DZx>P0`JtO0jF4|l_o8dIo=qhSp~RQZG~y0O)uwL}2zG7VliJ@dxK{YQ8r53CPqO3r?>e($=~@5m(8K_+jPl1ud()a$qwk<w1q@gTe;yNwCZ#BDR;sxsDu0WMZVMVpG6)<_R_E-LpT3#QCs`L2Jny1x2}%!-Q|lWUKCVMWX1)z=qvZWL-mGPjn=rFt}U2G=h?gq3C&z+{9Ip?6`S((Ej*=a7^MvkRm`^+WpZAZuibW6n@ms!L2@9%Nd5txqQDXkTTQch^?lSzG-M7O4B_Bk%d?{+FeH^IEi-E!@RH+nY%x@`Jg=0RbtZo4zi=h8gv?N<0?*#8}qBSZ=L^Y;OgDgI9l}ypNd7)Ubw~HjjmPtIb-`dfL-u8EfA_`lV^4u=DWYy|3>n$m|=_j_t-b;nUQB&no2j@+TMVtRrN&!Q^!$Zb_!B?=F7FSkqdVIPM3iH&SQnSsiM|2Xmp!TF=(`Qz6};YKaR3O0Za}qpx5al@&#uTm<_l`0ZVOvn^~4`~ad)(snutZ8|xeo}N_o0Mot37Ca1u7}xtpbrf?5!*}je1)T^KpIKt4mse6sL{wKbY_Js+jEij6_jcJLGZ{&8E+PZr{)^1@^dfvc`uOw$1I&Un^8Ov~`TNM)?tKoYIJOrsyv*Hiy!$t|E`a<vKGzST<C%_$Zj+{yC!T-ls3>us{2O_vls-nk&(Irh5ohITZrTZz&aL=0%kPFk8-<F5*A;BnuP;il4F=BW%<$-s%y-y(_C81`VPs>L>G|>}5iSI_A$%T+hmN3KS1Y&c5miacsMhw?juEYZfT(Q*J%Scyg8+>YVcs}s{1{SOTTqSP!m;`2k?F9(Mw6&n{VHjJV+2LD8dBd3O^0?M?)>YGMQKFCSvZtvZJ;2cI#~&zZf7Az^`-eC1nyg=Bx6bEuzS4NEN`)8lNy=nc7=XHQ_h+<>W#=EW*;`)VMKol`RKu8k*ruQ*#yS(Qr?MSj|STvc6gzxrk@UAdvS5jm|gG-+sX12m|5JUsRZKTyxFI|&~3N5fS!`%8u$uGd74zvZgy%V{?b?dh&-jf5i?AWM;Z=PTJ1Todpbxw*%<RD*>6+T;eXKdG!IerC3uGkWmXA)4hhXD(a(LJeSVb1y<3Q~sKi;+H4jYbXgF_6n}->6F-jvTgCSUg5O52T2wy<qQJJx*uUG_n0APFc0U5l3TiwEj*+WVffS5KkTw->UVlva=QMv}5$dG?*emH`5Gp6)w&fAnJaK^z_Ivj%A_eJJwJ#EOC29C92Fhqq85~eOrC%Ms4QA{&^oEbPo={h1FNN6CCJS$1q=%Zc{pDoqY&<gqqB~msJ6U`)jd-lft+w1q=U%&9j^{Z#Oi<D)uZ6nN()SaQ@_yP}*DWg{f2h1-gahOGi7%;rl^F6|#F<4G3#lc9`Jj4%1VC&xcn#ICgdvp)4o1K4mZS`yK@ptGcwS0Qx#=9HeKk&|e>plIm_xnre$HZbV+2-l2J-xN|Xhn}NDm~5Q*F%su5WlHyjRGjp%m}6-7esioQP89doA2M>{Oki515S<P-R}3_uRZ>A7$!(GSYP>4wXMTR2WGvta+y2iZs)85PC6Fsh&Y9`+%^#QVy6m+5qYx#k0cS-TLUE7nt_bl=AF>J#O?J?$R0900GzN}6|An}8ejx6(XYLl00#K2JzD;Mm*2pkf#+{{7q82sr1xZH{q%X%b6>h@hwBzZ^xb=Q?LD&h;DH`ex>$lx5tv+m(Yr28qNbA)gL&ln2%#Qcp`SA$LvSv?3w+Nd)3ecsAOZQc)yv#LoJcqe(1>m@U#@Q6_}nCN74^bwzZRAdGaI|p;p2OJOH$N9-<#i;qW<2u2?7F4k4*uq!<e+6rkyiyRlYRa2(BR;Nj4JQN|1h~ofa@HtR?sN60NvxC8|YWF=b<mV?e#Jq;(NUT*xt?KGU*_vVY&D39L*|tP0*9`ELULsi4<Es20z>^H2VSYA7&;8C$#zb?G5Q!8VAV&fbMfER$3B{B`a{PHI2kt~@WKHS@7_ptD?F3}%N^eOAJq9s5rc8Enh*M*2xh@!KGqWAZBI2`3Uh(Iu#m%bPSqKT%mYRQ1&&{gu8tX=8rIRc-l^VLG<!P>qMg=Y1>{v^SYYjR=ywC2lG&;p}0)pJa2%%baKwY9N%OnP>GaY1p9ynUMXY`0JEFsr*R`Bu>#CI|jD=Nfn5_b%e$p_^6xb3~PC=Yr2At_QI1vZ>m$FL6;40+`O&yB_|!`jli9xd7#Pu5<DwpB+T$_BO`hOX4}@xS*~BQ!6bj(fvK<$xU?-vI~1q=7Zd54xv^NiO_pT2+iZZhVLKZm&>(JuxbxIIeTAvHyVo9jS3a;c!eAM0RRqfjSz&Vq<hgIFt>_C`!(PV6ehrYec0B%ewDnQd320FaDL2VfNYIMjrnrEZP1*PaeF(Fa1@z*oaW2`f<yp}(M<cbgH}?t_xT{{Rx+Q+a%i+x!m1`+ZcDoNc2M!($i-=~r<ngdG#;O};Yw64!>&x&R51&K803W=b{Q|p~d1jGEi+BDe{hxA10H?R;$hhM&u(o2m-DFcjK1%zC4-h#kBLJ5D;Cp0QlDacOD{v)vd0)H3!K&E=14E0T0vm5w!g|Ckn>Vh+X7-++*}T09$0!a_)SQ4+bj9#a!{&HzFxXkbiP%=SL)aw>WnXaMv9Mr!_8&U>9}52An141k;6(W>_D7HYuC(YyJRxlP49Y>mr6a%4pagpj-(@-i#e2vJJtZb~`;$8>09o8~;Ap4<qAw4HSv46wP)~LAec2cDC-o_w>Q`XvDKrVaLif=b-|@}&Nql;C?a`%m_3pY|r(?m;po_V|lf|}>7N=l6fjydJLHTLvdd&IjGOa6KjlcG1c4vEYd3EF2yWUxJVz3yDbbJeh;I#>6?3^nWnU1-RPTT`A0PRuFigf>yQ3xUjMF&$_S7s|EXYX#3s@Pp*yypzWxLTjVV4o4r;)9OR<tjIuNMC#Tj(6n`sOx=#Sv|TJ&tWL8y}K0+t_Ex=!ZNPHz|eFlTj_!n(d8!0+lAG92*-0&ZZ9l0;{Gq8n=u@@_R3RIp6QIbBGWU=`^;Nls&H#&=xrh<PN!DEU<DWvDB+}%BXNINc=h?*6(^JW`EWM%b0aX^H+Q|W7lD~-x8}U_?{D0`8-96XQZR&RF+{yrj399D4)p3@cMSK@wcIM4R3R_pEK<er8y2;`4vXv*Ytj<i`Os2FB8<WjE!FxyS^9o8DrTSxz*?7MfT*otN-mk1wKnk&tqLZ|^Q+n*u3|MyigWI1_q~T3U%bI@nu`<}-u;i)FJn?%<~+@!dTc!XV&mCa@9}rSq31jteDFTt^ABOiiafch?u{8&WHOb6GviXEFFzrx1FNn*QwDLmk>Q8WO(#>NrlaTRRDKjT-*xsEYx4lQVoi80ia3MpB1|iSLdhWfd8atjiqVK!0CE^A{1Jz9vi~A2UZhoHtnLjVd))5Cc(^*2+`*q~JV5r9F8nw#e0f4s4~zFA2F6yB52thKkz6J+Xm%zyvLl!MDU%Ne^1EBY5WFtZwiMe%WLupQDi28D^zSqA1(ksT@aEI-@k&c%Am#@t<XFBi%L4f$xA^gyiD8|GQbT=ZA;tq!3Mlxr784p!LdEzjyMmE*=9-oAc8s17`@cuC6!{|FBObI&KCG2)HK%ZRMXq!YM1*#k%MJbxxWuTkmVlQf7XdAnct15zfde*XcpY@`uLNc8!C%3KO&3+wC<3v-3o`OHHkL0L8EDrr0C8#F(IT$(Z5v{iYv#bgq?RowF*VYB=_ds4Ve{goGJ-<xq|bvG=WhalMti=}5~ROZR9bbUT5yK|!bM~+v@DsB=M1(%P@t(NbaL7Ea&<3W06~S_WZHU6JH>_<l>vfL7r{N*%p0VxaT`vI?q9|uAZtmb$zf(Qb{c>SrVkX|85Lj72eJ{`77I65m>StweKm)0A4}nVf5$uXefRp~?l<pk(U$suB#ECwk}z$fp%&^g#?<d=2j$!*XAh|%`HHlG>ii?j9i&Ho65hE=cjO0@LTf91L&$bUu^FDjYFg$&Z`tL*uvCFqtj75><g-9_lRlH1jO5apFgY!~!^w=GoR-f16v<5%DE&n5NN)Y)G!dQ#MkuzeBr(y$@I(lUJco&yvZP9Kqn@z@CxvPb=@_AABf*y%SvRtg;Cqoy8q++0P%4}gh8K#*<n5WCI28#s6<Ad3EQm-@=~N)kNRu##zfMS{z=JEu6|X<^KKq!R>);%yM?s+0l&F2NSd~49)|)r4dFMVwDd=~f@Yqe)AF`faFkOi`CrSmO7g>{#G#TgeAFQ>+L}X1-&~MZafgY;IGsm`I@)1Q{q{2kHN-#ya5~IhQscEqgIB0|x=fyE}`Grp7wh#nj;FF@#G-W&Fd#`Wi4U6o(`_y~)ZdaW0|4um~q(4$XBl|xXbiw`c4@h0XwyA;d{%BBjV<6|o^K0IRALBc!#9pHXZtw1Z#_-1{tq}Pies0~UKMRVh;h6|*aj$%e5I_lbj~f<M-bxWmjY3%^z9kG=v<LQ=$g{|lsDTW!3TTA>bFr_`{%X{HiN=H@VjHN=ZMtT~qM8W0Nf|-dQzF#*NIy~Bx7OHiJFO-d765-;z*tT!5fWluKF5=~Y!(mntDJUx4a4T@2OIZ4XUhGxM=Q+hEU*$Uu=xXq%AYQ#di35$22Y>m+nAflZu*smo%JtQy|?e8N3I|S-jxSyt9Ld&`&0MRbLfYD{?pAHpQHEP{fig~mXse_4+JRwiPbOHSMPcEKJae65TQ`7hFRST7MTHui;*>ELf5$lcP0$s0!i&$9}k?H4Gi%Q#5rQ>3JmDp!u5^iGwWZhdQX<uFF)`;I~^YHJ-Ew**Db0R!VwW0ev8>Yx}V(Hc<|YlHCDzs(Q+tUL~66BdX*|`rMQ3U3lwnF#jPmFS)Z}y7kRAt9-pFxRhN|Q`f1&i+T#|DA)Je(B{V!zCrpQBDX~|fqkG}lLSx=nPmIcIp)_^a;t@N14<D}*fmiA$X59917t>Q6uRDS+qN+mC1UBbd)|@9fXYxeCS(*;bSYfZAP6x)(gwvN8m^?q3nqj-!lB)8E5tS3V7AN+Ld^*A}iSowA!V|~F9m@4eJNec_@55)^2iMk~z2iMxanken#yj)%=DCYTe%V-ff?3|7VQ>BVOYg}Qmecl!C+CkXXpsqo3Wp^f$K!Rf^_ym!GSlJ&SY}%I$GA@+a&w($M=q^T1!9E-%37m-WxKLA){NpC=f3eCz6(b?U>KO&>y78$nM<lz&bQk_@kA`mxH*lx<W{ON<J<9gTvc~)%V>nbO4RVf&1yLicDt=ov(jLA_qyU!&cgPi^uqj<$f(Uu{Va=JUly;?TqYfzCU$g3E}PkM8iiik6W;=+hrsVAE<AJ57y1wm5hcBmx+@s?W?ZL8(Yk%FH3ew1Ro4{;u)WZ42k>1I4aGF!{HEh}+(ZD>39_oqRx5M!ZK6J{WN;w;+tQexAr-WZiBzj(7|XCA@QP?xuXdD(+X+?ZGw&q!u*C`6x!)Ona*rM!1?cr+(D6ZpOeqBHd1QzZui2EyLcmCyB{X-qB`-K3C+|evyfj)F>7B9w((LGTlsON_&KH;yalLDTJr08Xp@uAK1!gIm)#}F!vZ+Cn%xuJdW=ieOS3!Ca+Jm&x{gPm;TwM~SU<IKaRo&vE8~)lJmOMy~f&u2#vB^N!fzjz#Ih4B@W}<fu^RS*?0E7CuwiLZYK=D3&C3?5T&}QP!7KflS41!G{<e6e14Man;-rCBAFSiW-q+I<AMVyy3<5>_N)F=*WVkF-b6)LGo`Ame1CR9OjRirIaHc8C4D8-Xu=0UMpLKP!Ho{h{pFzM}Tck%vF!H(6v+KE*S_$KBUKMS5>cb3b*Q~2k|KJEJS%?2H*#ql?m%nhk;Ae%kaZ)A5v6YE}FELN%%BHNz|t-eg(z_xK<K8Lv>4yI9aPqZD`5uYNU1m*oEpLX~y-v{POuJ!VqqmYrnLDo%HTK<Tw5mY&fxB;Y5&AMXpIB<T~fqe0TlizLF47ItFb)-J)!+<X?X2mU`akY5Nh4-_88Eej-wpEC0n83v)Yg$8XcD-(RGP9!a0SsKMWH|-=Dm=#kWtj?8mPSp*1u$5N7^`1N=d;>rnd>mvYV@fYQgZwe=y@Jva{LI}S;31Op%}*ottnTuAgR5H%Wh%Y{uk#JCTbj&x}knZObLj#IQ9shk=7gn8X4u$>INVWI^t9oYh>6NvXN%z$8Amqj%Zk%=EZ4V93U4W?p2>$LJ^=@ws?JUkPcSbE%hLtjBq$$GJy#ai|Ck$kPhQfM?l3F!50|gTUc<O#;xD=4c9|SeyoxSpGm6*B)%kSMY&9I)nFlrom$f*f$`o{#3GdkZg7W%U{PDEo@&7kl(*K)^abNGx>6Q$Z=JR{*7p^x$_x9Ied$vn9|F~UiC>PzFA9cv;wwV|M)=>sF6E+NHUCM!*dD2I(A_Y^rZThmz}wzRg{Vx7nkoZN<GwnBdn>R@R7$3Kyfse5T20WX=@x8>6t|Faub1aVkL-AXtazw~c%hVdPQN+vEV_?yZ;<^2*A{U-B}G$($znKR!#|`_g`McOq=U%b8ax-PmES4?sPq6i+ANlisp};Kec-I*Uuq}_hWU8-nub@C?t}u&zC$7DSK4l)r&kH%fITyZ_3+Xlzmrf!0z1H}?hpzZ2&agUw#>mp+5`O;I9E)^5OnvjAGM{Ug#as>pWZZYw!A<JF54Y$W+Y_!kd*bwINguGv&AnrCLK7;pN!6_<&!zePc@Q2Mew)pY&>|+yS382p+>Y%Io(?qoDm!dre*a$x$Hf9bMy4u-o+KXmHy#hz0beIYgUx;q~EVnva58W-|Cd{D&4!b;VMIBw-68oCHX-^sTT#)K_wz9rpE$;9GxMH{7208r)(=bs3nACidlY%MkL)-D=X5opn{><y#|fkUaFBZB*F*FEEE`dJS4N2b`vX*8H=4_dD^u>p1M-YoglI%*!Q7d)qC)G<IlIvOJ>|*N08Il7Gy^Gk^Y>B#d~$|mtZz5S{us%fevuD@2D61&GA{Cmy8fLA*qfm+4Ph=MbUmuI#LEbSPJZxgm6s?%MPdLh{LIEz|<Tv&q@B>X*JE^Ui;N#yj^Ma!wrrt-%}WlTL)*oSLly$QpqRouPoa-WcARNc2GH$v50Zr`47F-6>fE*2AIt3@ps*e7g&TFb_oP8L8>8|eEDNg81EjPS^wjTcl9kjMxjjH;X76h>IZKEx_<f&NzSNE!jEI1Bd7TdvYSi(7mU<ok{LZv)1VfNLH43tGZ>7FxG?C1cyGzR;?{nIL!Ec+0W1>NB<<aecZZ+7<9+caOg&XdCFoZgf29klWHM>%N{_1JE>cd4!pX`hU#hY+(C!|~(*tuiV@fGoQ}uiPb>y&``YmN)0~&O!n_|%EcWYQ~+J^YFx1>v%dTrdQTVngfyY!we6>;isC@Ho~nbdp2wr)tZ7)_3hi{6P|v@)6=MGM5tS5&=LKCPMODs)8Zmvix?n>@woR@CjijVOJsZTI_C%qUO8!1(>_UcM4g*jU{_T(U5rC~f++-YiC#ZWjBq+mIznp4dWByocjae=}646|f~#s|95;hPx9gC~fXe+WbRu_%bL8wlLj4s<~P=-L0tQ43&HzU=1;pF!y8bfk;WT{OCUhq6wI|Hk*h+_bfZoVj-E<f}pc|nnG4U)oi%iLSUi^!+yanU50~LA*!9?!Nc7M5~>~(2PB<tf))ImP^EgO)>eX_TXy)xbKSo_<I4{q2btJvgh_IkaA*W(kWj~*FvplM#}v0%G+x}>mHb>)-t)VQT67qt>a9x{c$x^>+Ugm8W>XWlF)vSyzWl{%_q{)ejA|FQ;?epd+4g*WAZ!yppogcul`mc(uT9^<IL!#0W@-n0XGa*~S}|)Bz<}k_Y?iZUo08GsyB|F8u72TtbhW#B+ZGjFI_?UISf!CK!!Fn2Vf0MT7(SLv5)Swmy6^mP{loh}mb_aJSt=s$^S6LvtzY{y=3Zc7eC$z&toxq4&@+@V2TrBkuWwV=sNC^BT+vu9#T)cLeG#=hRAuA*SL-X!Z9~_c4f*8-akbSq4fV(**V|MPB@P8<+&fCV^}xr7{%<;!CI&;Qxy`vH!_je@SZwwd$V{!(Ss-mU22dgG-m!(KB_2sSERqfjT^fOIt`@}SWpiXd0?kHY46E<LvN94CHU=}8oZ>&}OC6Ir1O2BXPUC9B&<@S*WKWZ5-HXF{b2CP@)ZDIKvNR2cAf^gq{6OUZVSprG|7JVL<a=$~AOj3(t8GG+M~1}EG*(u7?sKCI3RZUYGT!hwx8g*owZqSg+OFjh)m`fcXW-~~j1DZ1mhluQ>=gJ+y73oFaH=Mt4dLf~v&0uus08>|s0)NZCbtozFayZcMdED{3&QZ}CM;zX$L7u_Sfn<<7|$l=Hc+g}<EQlCq=Vj>uQ5Cmg?3`Hbobgb@77(uH2lTN@7EsPQ}2@n`N9i&QE*wFa+<6?`h5M($K9{L>^|f!@gah|>2op6JTA6v^ZhM`nYRTLX|iJksC)@h0<7;)DeJ2@0(Ktn>=_w&++pF!BX3j~UuN;l{k?&#oV@vAJ;DbP6zBTHWP4ck{t%g1oD!~_uFd!{%5!0{Ila$5_TK)~L}`7CsGL)w>}(k+l7-;bc$s0F4#lt!W9pdi&VVT2y|wIp{w34T_A`P!?$MZ5mJnmI#p8>I<i-sd7Xt<03W%>jjl0(#W40QK`)+L!kS%_Zj~Y!b+}h?}a}+DdzS^%ofL~nf>%Le&ZaQK#s!`Z_d8?1y0MwVbSlcuD$YOsIY`|G&F~U=ZHBfa<tc6-)ZKC_YEQUM4g!|(x>drXGNhpwm(9-GmwEe#PY!H)nv^g0Sft!0oCPM`s)7ep>ACWBtUN=f@)=nviZ0IZUSUWw=ZdA)aux8dl?-<UOurOBQ8mWU1ww(?QIwLfW%vrAAy}Nnt()!c;TQQQpSgeMiEmyaZE0t8UTReoj_0neMLKOGmcR`{~IFr+1^~{?oeH>9D>IE;D^m0fea!=l96l+BnLVO9Nd-rztBRKfW9P9<+(0dApcOKp26GrYml1?SZ*zGCBYedu_PmgSJB$F@<=4nwQ6ILTLRbw7@je50GTH5}<@S51p<He>c6<UYr4V#h8V~w>;)?8?$9je$1;R1Gmy*Ikm`=Co|g(7}Fw4cxs>P~3Fk3%#&BK<n1Y9w%Al+<f?Da=u)DSh$ka7e4Sl5TCW(xe28`wqVP+Tqs<J6}6`aPNV=M_EGisexGvBR#;5MoPpaiW!7o;156!Tb0tTdTq8c7sGN2i;Irtc~N8UfmaSD#4C0)GH(fw`|}tGo~`1X8*foQH&&nJF=)2(lcGe#-NmktwBcm8@NcXR#UKjSd8JZyMGas898A8QB?%QV7C+@R11?KBQwIg%8)Qe=K^`s&1LO04L6+D@WiiS@R}k~zIagr~p5N}i@nGZf_uiH34i>0tT%Je~i}@bnN?&#iV#hHS3F+(G-WxZ0mdVZ2PdC1JgZYN2u^SvLrR0R(`G-m$SdOjlJoV0f3WeNack`|8=`-HhZ=KCI|FH4=2E2grc*{4uFWy{x`kvZ7R%Ks6ed5hGWfH_6p0I1aAXT$QbfRKe5m5g}g_v^i{8jJmPgps_CM4#KW<^B7CV{NeqI}9S77D$OVeAWLjd56|X#%c)X66SDqRb?8`C`PhML6TVnlXw=9u<iOFivtVGEnm55dAqceJar-_%hZFaP6>mN^s@LAw|X>nx>3));4-<IM!4zgRdDumh~mjP1+I5@Ve&jX)c8#Ci8VreNMKx7XGKm<H5Qb++3PJxyaLeGY%F8Tq3+wS>dua*eE)xXTSH(t!zBFV|9x$6+J$rDEdLjC6=57nkJ7es(5RN5jHp}uu^e%a;Q><IprcU%@YPkA#ea0B`${Ip(QH=v@#58Q6CFz5$eFUaYAyes|d~4w+A*K(sI_A*xaDJ7%Dc)T!vdj<wH!aq9i|2RHe~_sL1pmtg3hltIo0+Q(g9FSjA$AJW8-4`_r~3lnc{Zz#vi?F(<P-PuCv5<6XThlghGsz<s!t4EP4x%7_iDt(;XQH03>8$To^ul2L=GqBwQ+I$OZkeAibc>P@ef@D;~Tb{i9k*z=>Al=MttK2)<6iv8pxcq>_IEe>(?ky?$~Vv?O$_zN!N<-_H>n0m>-R<4lIS<eg&Vh{``t;Pe;Qk!Iy7jsS5EyFRjS)V~YY{IybRI0nPv-ynnqEv_Zw8$jIUi<B|JX2yLfw)95JK}mVzn8#|g#+<E8UZS@J1HmGjvJD^%d93k7uOm5W$+sZGI3gaI?!2|0gSFI=2`56uIsXX{PLmccw`UEyI*h3D0;A^P3%!$hp^;ODE7NGq{Tu_hR~}SlEb#|{FqKSup{C*)asBbll8@=S$O3zoiYCs<zGPY5(vTC^ZVUvkAoa@3a<}2F9CHL)G<Q?^^v1{4ml^=r!)pCx)?NpA+Q_~Brl1$C3UN2A!HP=WoQBYQhtbof*5S?gd^5Pw;!<!v^*Pj0*G020k6NgWfmGf;c%Bgq!46};U9=w%}^YWPFeugupMprYt~H28xOu%U*$2_nA~H2M=k^`*W}LGwg?{%OEOJ&VG(K}dsl1!E50m##^nCCt=7paJjj^7`*DtH__0&`D5f>H3J7MM$Sv4oBWpLl%fXA@3f>=)86sHBf0z(39x^E;wTVk3rb&VQ%@ab40aAhzJFU*bLb18jKSf5>kPPUi!y4(F32z&TJS?*t{5XRB!bFqw7kKpyHNAxbvlSI6cW|K~ZABK5L)99Ii))Sy5yfB36k9IW->_FqMKe<pYMGtC+?BvM>g^_Mb0&+a*93nF?K$^{nsLi5L1tctv;X>gchJX%eE(IuhkniOU(IhxpTUx_I4ZO&!orAT;LIIv6h!F=oz_}yErKy86`X+OC@?!F$3wd-E1VAh0OElP<1qbNPsN<F1_J6nSXE%#HFPn*vJL)BW5Rr^Sf(My?124kJbxFBq~}(=EAN^v3@;K=J5=fm5%<D{y)bXeNPBST=b2%LOzOe&n|I{_04Ed<ws9*CpXz_>AKveNdM>O3Wro~J9_)dw5?dMl)ZYE#&GpOYet2^J<unGX^Pap1$h~><O83@EclpcS?vtUgp_oS@?*gHFw40skqft|tD^ygT^yGRj=4vej_KmvDVKYWyfr{1}w=Zt~@jC;#ZL1O3$ijAj{J@?n)2}>3Lsjx~XfL=Shn5!@6UZ(G3s)|qMXyU}Oy!8%XnYExso<^m(0KL_*_H?Q>=Cb1k7}9cNj{BO!3n#ECJ&~Nk8*+RaX$@XTk~<h>_gsSu2)xuw+N}kAWV0G0`@^sLGxjM0e<B3nuzqjnh+m24ip;9Z>`@mfs#2^?ICAo{``Nw9cKF!gJ2R#)QAcPx)Oc5z7Ug9pj6ss(U1{lGX!So&cimo-10{Fn`+CA?w60NG+(cXU@W0KD}YQK;<VMn93IfPf+jjLm5_NSSV#X3#ns>=4DoKzD+g$7z-k@LUMi5umqMZ}bb1I5Guj5}b%v>>8XSSv3MKr8kVm9s5E>7IB4O;Ki#bt=PkhNHRHu~tN#qaJ#m+8@m={EX+?x5cS+!7MHB5Dst=>;3<FH-JtaJTIG0ef8dA;`~wn+J8?a2pHDADVyP+7ls!FzUwbL5XsZ=SnkOeO%I?mKt9PrvP+{^R=7n`R5ETtiThK>3GDGr`D*G!dxgKbVr3F=-sKQM84y<NM(l=ulFJ*<#7vN~$~?`_j`?`G3KFo>uKTVLVD<Hc|Y^^qCCd4_3bLKZA59wm`f4Uh7?Tpv=|!T&1*ybq)ekm9F)$pS`1^@0zdITdo=DcmW9WLa{1iyq6tv$^pD?S;THtd*_~LRO`*6F7a99j9>JUGdNH!a)3c0bcI@^<S&3TZN{ZXRjO6s?f;;FpxM$v!I(zwMUMb7*G@3``>_0vtWb2QVg@g+9``F4;p=7kz#n&tV84FEsUgub!M!raOp?2dcQ5~L^Mg+qvkILfldGyOE<WlGt!{6I3Y<G_<MlP3lHJjK(L?S4THHt7MeZ;{)^_=6-^twWW4@?5nL|sU$K<6vkkI_i4Xucod%$fj!fsN70AZj)1cEvEc*02sfDU^c2$0KS6+>emlG_7)M=sb%Qj-thptAAw!TQP<-H*_dCv@c$?%Hr`#VVgcdmbI9V*Y4evW#8({wREZd6lSI=*O5hBH!H>O!v1+@ZC!RF=CqzywD~AnPQt|)_okQ{lUNNlc}&%HJ_y4=tmu_{S!<J&Io`}Y+?F8xYTRaCB^gRh&;ClJ?cTMfVUPaWeg{WG3rF9X!;gz>p9?NfxvOeA9?06jyX7vfAUk#|G=Zk+Ugx%;B>D)_MTjkXBF?^6(H~4`S&d+mVd@G#Xq_1Z-VSqypv18kNKeh85M?v4BE@KuJrml9w_4f=Opoyoh14=IWT%`Jvnq1tb{ncFV2-6x6!Ip>$Txld#UQGX9>sP%>TF(1x$!>E{kxlb5E7NSW~h6Jw^W)lWYxf!EgAoKW~e)pu$A92@@(bumrF3xK-`L`F!M;jV{CTKn^<XiaSjH2|!tLe9fKW0?1I<|1cZ9jqF>o#P}?;#5b#GmbX~q`3ts-k*w8av}7hV?4&ol*$Z15qDrd-;}8@W_oDCv6h&y*?lY*ZX}Z<8Sm5ST(3<!oIHW%&1_eGMHU}oVAU6o)C^6~1F@#BJ%I_6Yyb+pi{hvbruZP^`cvGLjLlleZ>(&(H;ZkU=v(Q*_iY=$6<@!G8H``MS!>l7qt+u-`OxIzY*~$sGJY0sOOAYz1qkP!mA7-s$JLiC`&XFRBPQAWSq2<TD?<=~pDnuGVie%tn4g2&0eUnQ;B*zsFu@i|U%t))K7pTad{RU=^8M2H%UG;)p(VzINTLcW#sP&+<z?Fd2y#x~~)ffCnnNo-=0$$=rEAgn&D|YA-C-Rq^;v#gz?s$_P<|U$wdI{E1;zosMu%d~M2cp};g7bGKSY9&VE^^UoeQMbtYs&ua&wk3VR7wi`0vIOBktyQaK<_x3wpdW{QoAm))Z>49F$GLeFjdpAHHvsqvgQ2NZPtey#d6s#C!=(j0F(d&4-==1h*Z80KZ{z}fAg)ohVnE%yc&u>W|E@@(NA6h({fN)SJoSdB&7~)!P!a^&in!R868D>QfO+wnO4$0(WZcerYbcaQB~Qmw)z%xvC@1pmBf^E>Oa%*t*%}sX{1ZsN^||bQKZZkazP(f6WtS0jzJM9V-bDAf+6)4(1lf0?wC}{?hAIKk~}V7M$45dQ4C!BrSl+D+H=quSQd7fnL~}M?UiJ;Zmn6zjBNvy4+2GGUA3=;UBf1rk2fT^ghtPm5ScFkdwyx~#o;cs$p(JeLYd6ry>xy!NB}JhEQ-mR0_$S%BO{Ff6S%~!K9wm2uZjEh6jl|Wu>K878Ed6wSrwYUssy(MSHk>neMVpjuxQI7vsL=ry{r6!0aO2PJiqF_{oxN!&aXYXwD#<yjc>2>$a~-arF-iF^U(}A+*ly{5}Q!3U|le$`C_ZshDCK3QQ2<=Kb%TKCTgwj2RBYy-^aH&5>*!IV<bv->qwOMfstsE6eC02V(}yPDBT*MK+3?5O05^{$CTWxW&@^5s5O!cZm|}#LqBc>zJ1>zi*#f<><D|Az~kqC0z5W@#@5VJ{ubs@`3FwJ5PY`=Vim1=jZOVX-yrTA-{i6C1$$71ps@KX>^h!M;w+JIl#!``3Da+_azNP}(}CaJT)x(`wLx832ZKBk80N*0mzN_tQFppB1m13WwP#a7R(azvQXcHY*vJXp2>kdP@1t{UD&B=Zu06fMZ0dGxS!g$WEbE`#V_nI-<!q0#4Qk~h#vZVaj>g4e6x&9p%LQD60vo2-Y!;VdfL#fv++J$9c_<@@j$|~|SG!rPwJ?5$+bjSc;w6u|uB(9OlAm^qo|3g1DKCWS4d^Y}oAJ&>wNi6iBbXX|0pvBRU8ok9(72~uqj-fM#P^6VGA&5tt^wJXA1jJgcJQRIA(3sETE%ZCn#F3h<o<fOQ<bM%i>{+wKm}7ZftGbtZ06G`Eqq?9)4=#3N!d)j%0<4-1S>qWwqm>w*{fQ&-jbDz>_%itel(u!)LOskxbAP=SURo-%U+a?+!KwM$y)W!<Ww?oXPXlnX7AC|&i(@W)2cUPFu@d1$9UY?ZW*UGy~!M$(sH@Xv>ZhXxCc&U8;&hdH?7qz0Yq$dNkIYh6qq;Q<4k9EmLj1vT=b4;xC$&%aV;VuU@+cQ<STa~u4#Bq<eb<q6VB0uvpbl;@Px3}vvKB}6G_xl0I;X~Ldv5qqBWo-$7x5w={|5%G(YO^L)k&hx>aoVi4N}@o^4iuiBy-IwipoCb~$oeGb`W>zZ}?VcghvYT+jO0DPBlzqA$G<J3=4nP%`DZ??E~2Lx2z_#}fK(cV0dG8vLuw%Wol=CTWn7wL?XK#*et=q$*rnz$?`7YeMUX`mTIBGoG+rR^`f4^M-1t<q8FE{%Et~TH8v98;wPIfe7%_fI@2#pP@a*#;f5u_rKuJxn_s|rB&3w2vAjB1v-WKO1bRTLgr?HzT?z5LC&jb&T55uw^(jzm;GmFsoP9;%sTC&zl!F}l~;KT98s*v39|>9CN23u(61ptSc9vpEvm)>VVC<l(LlpwYrfbpwX8>>=26pzqV$gj6y-6XgH@0Oswz*Wokhh5C$Hm&QC9=53eP^YVunidl_q+c?J<YcvoT1~%3Q2z#O0)cpB{06G}kbUf<=&r_xmb%w<N|F$5_FHy$Ry!0YKkaAC(Wg`(BI1gWiS_8pz-z_Lgx9?WI`UQnONPblPZ^{xz!{IioGEQzq%t(fvqAx3*9QYUL$mvCUXkCW0NDxC%3S-$bhV?HWjyEms-=6IpFug_~t<Ifd(|>P)c>@+N7-1-TAA=!MNqBfhpdg5aWTcZ+bT4N^*vN4YbjrVwP^FoMjWr3wP<E$srE+ad#UFPPMju;)2_xB2l_77DYjTjnBEo>Bd+1DC!9<BkM>UkDQffo1j@=xjsjEC@a)7+y3s1gJx^SZTS=>!`cl(`<q`Haj)&TJ2b^e!QmRP0l%~J)0pS`q=&>N#|gv4Lid5H&~`EG-XCWY8s?WIwD%hQ{)E*omUuA^iek`g#%s8ae4nJoiE5ENiB%5EK*}GwQ~X+@9liV(Z%9wWHM@Ftxw1VcO2p1n-tI9a&WMqifrz)2<kz{0yainTt=H+$Y}edyC)J(L401{O|cU}f*00v6xEZnQz#$xZLJhJ1_1=+1Q0fTUA&Z%qcJv&W`2UH2T25u+vXeSib<ZfuJ8;voSuGR!cbI?HYd%mo0LIKuBL3_*gVtGPymr;)$VECfWv)p$w`l8Ml#uv)M(~kfAbsEyC^Bzv%{(EaC&5_hT%#N72ESUXZveAU){fJyWg{srTx&MgN}>#ItDE~F=FEO!~0a@zUq73<9NJxfjfOf*3(r@EDi?@w_2GITBAdBIhEd2>vNrE$5G<Py3MIjRg~=D(7|5XsztYiIx*=&-dR{!dfhGUL01U)je;m8MwcJ!v@6wCE4ffS=Bf?FbAL)Lm>BD*N7cgz58^=u>&FTdFs@LDV>q~ab<u^>Qlr?UrP+UQ_nv))-Fpv1WoomXyGmgo$f?@CgRhz|0fa3PmAA)e(Y?9XT1Ljj5u9W=>^~*u-X6uBq(s4h`)rL)SBt1r<Dr~I;5{fAOGzszi_WL4vZ{d>YiLLwGK8P_!;mQel)$Q*c@E`-Def)8k+$rf$fpv0`soK_^F_wz^DodTfd!fB0hRzhP~7gjeiuekDGW2n&*p27E^pkwx4wFM?fFODi*KwwTHZMKS91cib!6CLL&)RAcRId>b%uk_zIw52Y_Pr>!+s@!6G;UO#z%Za>RF6YRlqG>6Wt~o$EreVKp-^RS{F(8M5ToR-O(bd4#%PBjJ0|j$YG~e{+X|nVJM{5O;%wtF|}LfjO|0@Z#O^v;n5YI)V}-vcbIkg{!I+2eC8Y3Ymyd>xf8uYQM2V6>X?OQ^I!H2?hy{|1^<lMvsKx}`(RRe%N}DGVzpeH8eo@{NUICaMutgdaf7Ojj>7ur<5A1j<kCa(x0QymVt!j~y?{Zn6NGJ<#MuIP09zkrpUdvl43@RUJUYOs$t!WrdG<>QUMXUNyUk>UlJE}D;hp?eRBt%^s#{JvM=<f0!eEmMRQg!G(WG;@Q&Zaa*^)WL1;U47AUP9<+(xAZQ}8iNnoB7k4aMD6eMyebi_Kw#rDG1hOgLqW*=V}x^USSME(<G2c6vH3#x?2R4?h?c;7}KE^`XC%=Uq&urq$Snau%wmNFTt@0`T2drz-rXdp6;S(dv{+ZmWe-;vPk^;xX?0n0zW8$v}BHTo%(#4Et1AgFS5ee9{=YP=iM<IWnutp9slPwcc`hRJuqOf#$$+f*2n6!8#%SSV~Y@xD)Kx(iF&Ihf;}DA`3qgi7Y**6KQy!NbE@Lpyx~?!=ESUc_cBCNK^GhnhHh}qlt0$oT7p({5b{%6THG0JdINi>^<}h&(L3Dgr3LYDGR-h@h@V_aTvvpM23whg&!Qlf5v$iX&7ai8tq7ov64&z{y)x|@=7q!5gN%j?>Rl1fdB8H!SdO{sIyR##@^X*((t7*8ZHc*mB4!(a)yS$ir_2wp~R>JOd94p!A2@TlSw5q2&%M%MJAod(6<n%Y?v8<a0Y=v#jI!~0so(XN@JqZXd*L8Q^T%!SvHYDNRJ^rS;<%;GsdCMR}X-Q@ZJHXybjC*q3^>x10O<&P%XB`0E?7ZWd`4obOQbl@R!Rnl1Ypp#K+`JMgTWx7FirAD;iC}|Je%eNQ|;FShp-qctYURSYianJnDZA2DpP3LBMikJTbzs2tmw>pbskWaS;}TzLl1JkERo&X*PcjfzeE26v6MW%IHX9bcEJPEaoU+^(aAhf`4o@n;6Zq70+@^868WEj?w0jB{0Vwd;xf~@x<sjLtO#gQJClijYmKv3m{9;M8<iuEX*mL*rBXR7KTg+I3iadJCcC^k8|*{foEZOqqLy~OtOIMS!0L7sK;oJWkoai3?jUb6<OGn;{u%-;Ih!ggbv*?c#l;OHyH@V(upxz5J^GC0F5)WeZ*!RgU^hxxxhbcOk)7<QHDISApkPVceMB(p!*mbmFx`uK$wsE`)3TmIL=TGN6X4!MJ9;Sh&A2;?-9lEhfQz?44P2_tjLOB)C^h1_+ssV1sGu@gim<~5CH;!MBg3o6&(4PT&o@MCxRT;hL^xJcd!wTiy}aYaUwhdGj;%ePw)knQyGUADI%(4f`kI*q-knnVvWZE<e3EPjaPzMjcC+p97a7#Q<1b7)|$~2F}ZR0&KQALEXz2M8-|%;myg5N8t0SaUk8xl*38I$CIHMd2Or!MFp@OGCqd^XfD$q?F2FqjLncZg`Nsqdc~oK#gWLp6Axq>*Q1%JfgE;Ptk318w@&qBt>?Yt#<7`lpv`@gdCW!b-fTvUVK1HJv8#|rCp{My+Iaf$wqYO0`&`PIpm?MmT`Gq)vQGG9^Q@BC1d|!$$W4kd%2mRq7)^N<3kl)6MjMLWlKZCe4LHk<l^fY!O5lUcQnmB4&pd+6mGSzg3O<HhA<f`cr&OijGMaG&QB`g_}EE-0f&I*dn2Z_uzJ;sQ!{4`WZlQ@7F%*$Xh>2VDa(#T!Yj4#W{0DHxsF<N*&LS(R+G*RJkSr5g7j9|Q?B1|Pi1k{Hja9RA}>{zshsbq+7_@D(wOFt>vEdmh0Y}3SWGm?;^sULrsSi|QG3>SX@rExY~T!Fcar1V{!2Bw?lj2CGmFNB$lWVnF9Cxe`Kgrk|*-)ZE%Bcq%}ipDUR5ylXt%m>^Te;6|pUxrpAJ2)#5^<XL^#L4hOybQiJB50Ic7-YkvDdOpZmZgCY;|~+t_^^Q!<IfnAV|W=%Wt2!T>y?+H@HWb2M^O`MjAm*11$R#aOU9oa3R_MCQ^p?>>0}wqg|TKrdtL|=8D*SOvVb)3<}~qVg~g<SJ>yT74TzV)WU?gLDdHrsX!>C*%RvD=I!%09l~=PUjAm(Tp`5`BVLI6{VzFalUjnPfAJ$rwK|k4X;t75k%m$fu0&#`+2kaVuRQ{C)hD|>x+{pC@o{c|a>UKk}JvPF&ftWOM?Xgk8ro~)<apTVpffu|KWozs)<4gCzyzys(b|xP<a__Nm+D1NjfPLc+KM3%l0|&<+w&8ed<l)H5*?RHTz{T-roTkX91AH8RvZ|br23C$g6N;dh24;>w49~=PfS=<}meBxFh-U>Rj*JVC15d}FapH5LN0`hw;g~-ccutR}*}fD*gy-})6SF1ff#>x22<<|_Z}B-j&PF{ZAPdjwab42J=QQ#25ufnDb9#ISp_mT_C`gYpsnMVEI39Z@m@MlT;=zXwXg<9~RzD#n5?%&VnaB{$ljk^iPEU+zf+s$wCq}8Ir2WY7CzwPfH#$70C&q}lNo0iQH1T+4N#Hp>!8f{sp;$CA!Os|?IbdOmi;ZH(U_pi${fN)VuppDlh(b{lj&i9Hj*Q}z1_hbaC{0^XJ$%ljgq$Vn!8T2eaXDEwhQ&<n;CzrD*Wfurj6Ulh{NOp0njq4lG=|v-HZM327G%<CE(r4;(|9V)aAXu^FqiZQO>j)Mg{h?3u}B<C@HvytGT~-S7Q$T8V>BE=Pw+XD-ofY00S3>RG?&RacH?s<ElDt+9~5LV?1&^#8J{!6=f{<kGaj%rOk@#UD?=x2VvYWn@ra!v-lQC;@sv$`KI0!Uu^^KXB8#X9b0IcAp&a9(S!SG2PZEDD$Ydr2LWx3{3_D*dLI##(h|`ZMSc)YXV)dhvtmAWLgvl-bmqt)$Fv^&tEQ9%sWW|YG^a%4Ycs)L6Ms{#2!=VPx8RGS61#zJuGcv(R8DATC&Wxt``G<oSo-?Cdob<N>lw?LTOd%r{44yNigk#F?f`SY`f6G-IMHNVv3F{I|p&&C#OkL6*e9nySko%nj4N5YM-TRyZ3o@e<oRNxe!*sIzq#_p=3Nl%y7!x!JpEFs;QRLFVb0#}No88ArD9B_-1!nN+KtYDi<-W*_&*;HPBbQTzf=pKMQMpT?B$H*z7rC|JIg=FvGS@f3b7oA4W>QTAL*s{Zi5BpjA$G405(+Y7Y&zoDiq9E5tTR%>%V08NS-$Hys=#w*jL5Y@lEzR&wSzMdJ{5S*j4_3vzg$p|8Jp0=6nxI?NQounbAf`)jx<wwiMhaYW(N};{I!OH3?1ARE{6q~9is#ZAN!ynL&x_C$r<oDL(D!UPrdk@5l46l+a0KJ<H`pi?qNY@#{}OZqBYEBoKt6s0Z@<`PqXC`vH})l#xoqrg%%HHGd@D1om@hA&ft-rI1OJxc+Sv?URf=8&WtmWQZhMs&d{kor3?n9GR~GtvWjt3A5KUpikhfGq;rDb8lE!~X}<bmDo~Ii86e{e5Ll3z7~wi&zKP*EGr^>Nskwyb%tV%x4?aJ5&P<GP<(6y=lbPUz%kL3BI>FG)*L=ckCMGz<ICS87B*oy*$(hu+Yw<ybhK7DB4-e`*%*8{R7sDJpXdwS|-W}}HQqp>9Ni>PNQSlFT5{d4oRIFBKilt-nUevC6w{)yguhiPR_$%JU+|oj=*s63ElE<5P$JW0}g_m9TRTe7k{ykFES}u23H6gd}$kF{K$as_1%t(R$lnX?jhJ_a9nd($s9R~hH{yQ9DB0ODs_{hk%$%SL(N;5`==0g1#b82<i2U@~tDLd<{_W%E!Y_b6-o2>6-1G6tKwA9S(+(ibXE9qV`Qv=~49ig)32osFdB)niw{WG}tRQh%A!5|8~9DJZoZx<h@A0HoQP|c{zis`B;1@?4+hU!tVZ7x6LERe9r1+X}JBY&y*sU8{9D1JhOYHX<_HDjUWaS4P4o2=j%=qV%e$@S-1^a4&rL6@GG<`|;hnL<yT`b^98byqxe%vYsqr9n;7DLpF%mh&dYtGVLBLNT98PD~hnC;fT*Ah+MrzW&L*eOS2Q!ZDcK94;dHfGE&%+l3-@m^aF`t9z2m7udDQyq#GkoWUOaUyrVopTv`p1OJ#CaB)_77w~Y4TRe+m>WyBUaeiuGf3?9Q9S=mh7gd3WQm_IH+L||bYAovxkH)pTdbA(%qp688rI_8%pldQtHl5KLxQ%*g9<k1fSXv?M_2R-zncidrD2lrn-ej`cd{}I3C{L+(3KwTZv05AV^<1-9Ccix6orWrT?Mdy|ZO+tNu6BP>bI?*FEz?DVnL2DNq+xjssUrjfWcuwH=}xmnM=T-S^y>{&lJZ?a$bNu0L8(u@<XT)rj^X|$PdBsG#iLg!MdzjXp)NQ|292gWTR8z(*eNa)8jZTVYANO^ycCt;*rd0gen9-gfdTdBgmsjkUHWXpB1j-J7hHsmJ!SXbV;*|qe3VT3{#ne4jkan`4dtzMpEsVh7~u?_`AG{ZBm%23y>;M>C7mO%3=TO_Zpp=y%?;Nu7UhpAVL@(qVYMwzw|1{Hi<8{cz)^WL)jZ_jyv)uZmT0Z~MX_DrC!`lUOpgwy(!=SrlS${`|Dzc=1{7QEJ>?2;j#g9;q8iIK?jOsY^k^=-BR8s_&v#aug`>q26(H3jQ}1?~fj$9#2De2F<z@0<h~@AgU!Y)zCkSzT6o)ms0^Kr?mHJyimW}1p8#mrvdvwXWdU5^2`R=W+11HfE2CL@n_Q-Q3yif98M6i$g3I48Bukzsk@?{xs3|pi3Q{4U|od(IsuVC1cgmbjh!1Lv9ikjO)Sun`MPx3^o&c?UjufO^DhbI@;SI@&ASi!q+$$PfUCaI_OTw7V8fHZ?n_raC5M_;LLDddh;$A7u@=pJvt0^49>T}pn9)u^a@VcENNSBswSY&?AmBV_%-09mA$&E-$Ll`r7YQA5gLKRk9Yo*M9$(x9_`;j#DZjqa7J-j(a#xu?*JxEi^ghb?wCo_`2^dS|~`zjqy*Y&?Ciw)!2WQ9t)p_x#n(_nvR>e)!k!^@rW%FE{>rcYF8xIe3pn*B*OUFT*g_9<6$B-`zZS2`YJSe+pxULf-A4mp4z}W$m}~_-R)jz>6QAeBwv8Ew+Ef5IcrA!vEGISsXpG3sK48FX`ma0-UJ2&aXQ)Hf@Se-mbO^m2xl$h9!gxP@5cfN;eM==>ExpC<DSxU%joidkHvKrR40bp(JVKM51e3JZvHLjz=lVUV}?)q%}sdhl=RP><n%$Uw{?E6t|DQ)Z!ocy^^qudn7i=LgvqbsEmWH6j{=0AfnUGHl=GiHQ{ArzbNjQkp)VAI0X0r-}cG<ooc)COQ7G)rPydZ7-d$EqkxzFSUcF^v1<Kz-hpH6d}VH)^Vft!krwhXb#N022LxFbkO(iTHh}52#{yr2BWhdmKUddd6n5EHfLyR&a2GzkwEpxy-`x(|+m2ejduzqJrPN$|^f|QmKDvrIfz^r|Prra#Y}+}o2fNgYKdZTj<lc99dj<D`%Y)O=W6^9pzvg}Tv3LF>@5^@>b)cPeWAofa?~QXC_n-09-1KVm^wYIRD=Gy108cEeP;4SGDk7lyBf){#2f+4=_wXzykZY@#*H(Zk{?&<P6HYo42S|L~z4RG;cJ0wJm8TPa7I)j-+nRg4AT|LV;ncd;S~5J9B{-yy{0YZ$S6zg%BD^uKbE66naV{1cMuYBXJQ(apjzWztLj{8Ia>7AnD3ejntPIn)uuBM{W)Q61kKgiESEwnbHTm?3_vA`MS`+oNM$k>EHQ}l38*Uw#-_Uf<8HBY}BO~+Z0|8hm;do_4V%g1PWjg8ONFacoHZl>`QRxtxPo}0{s99QVW<o&Xh4vz%f&N9Y{G*BCble#X0!az$11BO{OAL1#273`bD`SSi6s>*0zP59J1XKq%Ke_?CdF}b_?i&w+UD5DpnIr%v(M}BiGl8sBpG>iUT4FHTB^R@9CZ|<RPunNN!W;9IQ`)O@0XZOIE<BxBUlYN)7}HuA7|e(TWak>rkVa{((THpmOyMP)=&%~aCas3+mP4wKXw|b=eHLSDSwCO}9d;8B$2bWg?S%?x%kHSu@Cuy1Jrc-xS%BrdvSbMdP8h9xDss0iLK&RzjSlq@MekSph{olJRDvmnWW52ecA&B|#>3yf3k=%3_;~ZorS9F^-pB92o?^o2*&CQPt~3v<{nF;G<&7^$6b9M;@@n_;9hCS%GQE6fbM=Fb`=7%f-uZ_<<>%zRKs~6GU)}LK59pgv8`&5El*L-fwGkZUBVn)cGrv|o!>bQX=WOv@*r$BzBx_Ipno%e};ax0Dk#B3-JCnTDR^>K<oSZ!k2Uf^QuEGl~^;JlU>c2MY0TuhyJ9Edo|9JD-4Mec>-#MEf{H1&EGraZu?5xB#Ko=%6g3xDJXxs%1s^5Y$oLi1DUV$NuweiN|iDT|ko+(tFV>x{`lgCW5C<aeq|M3d}sxIzd>R6ei;+Wsu==WGKIyk$x$-4VlK9G5UwR%Zjp^qNrv1wbd`>l@D3JcLrqzO_R2FQrs#@C$jQ!^e?(pt(jJkaQppSZTFcNGs@Y*pysNhJ+T1wY~#$<k;e$kb6LsW2i3Px@0#ptYsM())slG3@}4D-~;4)PW)rJJ4!3G}pu%&qKECX=0p{?0eIc!bzJO>`yUm_|sU)N+6!Pmf*}M**;iRn$LiWB`O3FXXI%>iU>N;;N0y{Dm~!8Lp+hd&@@jeU`xGVt2YU~O#7Jyh8Bl%Hgp!*1Ud~#p`1J>bBOBI+RQ?b{K-yQn@op6*l(ypmz|3Zq-BNioTZoHw6l%BH;;+zE;wg<APWI&Mm-5B+9pW_7m;yLmkLOcz~Uzvcg%aA>NPcrp*7(dYYYyomI~%gY+Kof3KH6pEO3~R!|;Kt4l3b9j>CFJggV+|+^Xggw5{)B4<*U7oizBV#5p;HLL5k}#N>x^AR-P4p;Rg2xa5liSS(5juA=l*w8WNB@~!AcYqwz}3<E?mvCL5P&eZEw@o%q`mrU=t@%;nu${#FBH+xD;0YZcR%a?W*U|(-@nZ#bi9Bs``t%P}}SO|lbjA0-&k7^tP7Jp^^@&oU)(`zfAudVz}i-<)XAPGZ%0)vK;QnU^j6dO8u3Y|KJl1y0{8zO%>R00phVY!bFB^=@Nl^^P~XNSk3q~;z+gL8gpD2{f7Sqp_c-mJ6<^Pux2dx&9x>`D<;`uQREM9HlVS!kc|QxT-zA|`B=*7u;=&X!5Gy?ZX+^<*m{iM~Y_Vr#z`Gl|<fW-GuKqJBJv9}1~DXvBy(vUh*ZIl1MLoN}h7Fd6sB{@`&+ORX7(5CS&F@c$9)fXC`n%3^$PlBn|E$|R5!c%Q$ue)TL_la{?lU!pegy@%aP_g{#8=n{h%ks(KLsuh@et$sX`<7nIfDkqH3g^|G^uV7YTQdnE8HR3y*bZRDK-vUgAb#f8@5eDE<Gytb3a_Jvo0H&TY87G~|jZEZ5#&u(EYpE6G2R&6|KBMSrNZIn1db_(yuUCwDhzYZd9^w?D2ah02tt5Vx!&W>BO%57vY|W*O!Df$+u*EXtZfm|X^Z&JXEk1G_W&U4j*|K)ljMr6N)vv*hNJx$##gL1|A!J*t>U!84?ZfP>9W38Uh&$p#5V+GF2Z(S4DToNTfG+5i2atcp>xKV<Usd&^x~FG$*Sn4#WaHqOp02N|zWQEO^*vVH8aXNw<<w_?sxPvGX9xLBTPI7m%)0UTZdSdy7har^ujf3wRQt{@HTJXEv2U+lughp(Y#m0Od#J4hQF)h4y*v2Dew)#9_ZZ7L`uMwYCkVH#N~l_|4CE;n8#k>>-E!bO<oowc8aK$xjT3=41Ug>)!3bAsYv0-VZT_ceH^W#Z2U-igo8M_V88D`<(8{=uPFK(Ne)qxs55K?vo1g4|>jQU>X3M_!L`#t2X;?(Ksu<Pa15@c@lliH|V`Z~Wm&Vmjs5|vaTJ+__hPc_ObaDV&K9NjW4H(3?Rld4nXN3RiabFT(&gt5QovF>VCT@;qnzR+?TFqbHT3c5we|<%4Jm;U-#@I&yR{P9?Y>q+kW3O7`BwVN2pY7pb`y|*#cSkv`YbRGjs%t6R_nNxO7n{d+Ou;Qp{dlEwT|;`;HO&xRvw&J^o4G?Qp});1S|aDc|NY{@o8JLqc5KG>w$5*~|IT~Xi`!l@|KgW>-~RUAyZ`goUw^M(Xr$vWd-w_?w%Iv5dUd1%BX_Dgt<SEnUEf4oC!!X4=&qb#4Uma*NV>eB9EdAmY&{kpx+=)TZB*M-n6oe}=K28Iq8;54%qA~yem_PD4p*ByYV#gfl{owtSBb-?R(>)oaV)5Wao8#xPfo=QV<E40N=_V5MWPdaGTpJl$)<@ODob)urHg^iPqN;{5k`8v(iqXm^3?KCv*g<`RTDw;QNP4(Q;woL_TDX5Sd60O;@(SP?SbZF@0|tJ6dt~Kz>O0u9uUlKn8zP$#()3bs+TOaESb$S?>V^%bbR=_03F_`sss3rj#nWdtoRo|dWf<Cj%=}*K{){XX!MQcZehx*mlsbgFaO+R`t^vaocc5Mc<o4DJY2&5xw-%AKEe_Ji~D$d>HI5mvSOw>rpaajUT9c88dK6+g`dAQ-Pzi3QgyK{IN)5n(B2w0v(j0oxzK-k|4;w1|IwQZT;`5$j)YNWPD)@HES)Lgr8i2gl+U-lsFqbL9vgt;h{gS~Q$yvb?73cY)(XVrEQ<$k(vG5~SUm11qm<8ALUcT%ZU)ObRtJOYxBLM76lQScDf`ff!oUuRp>8`n!8wb$SOwJ-F<l<KuWZBQx@KvmMLHfA_OQnmZ@TU>l@Glgy3&O|_hHx0i9zpYL-5vTF_2OLId<uD)S2LLi{|q{uj9FW`ehwQqZ^y!NNz57lt=?IXmG>Y<2N=}Zm;RrtjlfG@#Ma-*`747-vZD<M~{Y<0?O*XTgg=aQ!7NnBg5dZ_jC1W?^mDw(ifsjVrvt6WAk&(2`@WG@^dkZD4oRFB+2Y=VQ!4Q-@m^9)A#p&_kq1q{PVvy&d~cG{j7P2HZ$kF9xjh)HX#5Vch7sbriVBq?F7Jl`KGTa)=KJd!XScIwcKg%`v;`6JCi~KcdXRFvv;Z2s%VA5pa^H{1Wm2!9)4{?9DI<7jbC{8dk=p2hZ;5Ecx)tFcRU~kKOPKY&@KHC$e*ViYTKbR+_h<chC$5PYd+|9wB&BZ{JF|s-u?6bpWY3p==)t7+9351wN;(kRZrT3!BB*nEqiD-&`Pas+cW9W4CDc3sG7XGwYE9x0_-9>IK*<qfr&pHvk)J})?*eI#jy9i|BhbTx<1C&*KOFqN58!P{&)B8y><WnU;p)A-oZm4AA9OAe(4{+9PNMjtG$2wjlItL(J%Kt{?UW){O8^qzu$ZFod@5?^Z(!b!S6qNbrajY`yYM0|K`svEqwdE=H<Qr{qf$n-oeWemMhx(<2U!-{)7F|Ez;FZY=ZXR{n3N>emJ+83oN`o+wyZmw>S8k!(N#r>4dOZZk{<YUclKl?}JN>QHQ-%-z`-*CyC|+-_pyDyNr`t+sg5$JzeS?$L+oG!~O66_Aejbd+^cQd-p!{sPc%u|KYFp|Lf<?%$vf#+t;U^n=}hN8oA<MtILyURke${I(za5zOoFm<Xqxi+bc^OdO8)?yUuGBZMMcm^`NOP;b&V%);`2seRUc?*W=Ezs+;VF7`Ag`I1EZ&dFiAuGSxd(dX16IFA{LE{kpDi>-CFQPd)#_b6@#_J-hQ|GsJrE>yl-zrCOl+E8CjNRv>DPbM<YW;04%SOJ}}z?##xyGivqBi>qfYt)98O)Q04)_Jos@H0|gBVYxwty~C*At4gxeZ-z2{%cmuRn41QWI>~Z^Sm#&t#@20Z6^XdD-ZE4my|>*Pqkb6{p~CVmbH=~4avmgo?yg10uC0S^{bM@*hZ|xCNcS&9{~X<Qh@M<X-4`pTcPDpE6_$vPb|bbMor~^Pbg*kzDgfwc{o)L#hwu2M?5=zG`}WpOT~IlTE<HCBJb%RbdujNIqoBwBCy4-?`optEbVF>a^&Y)+5o52|raMra+k>-Sx`0i--h9T*e)2W^Id|#Ox#ypcUVL$~u`!uWS1w<^ynErie+vk?;D-B5g{0_0jvxIa`=h)g0L$U*!g=>)Qax)B*n94<Sv><%Yw$I0Yv4;=BzqLjN{9+mtu;_zXCFUMo=-SErk@1OqBx$=*oE95Do{Qc{*gQ@1}trx<-(q)ILCl_arKMOUGU#pkbv%z|9bc<%rL>4y?)$YZ=7}N8`x(rz4B5W80~7c{pl_1?)b_}QQ&~RFlShmTg*mS8{zM-N0*;tc2=9pUzs1&+A6DF;;cGPR(-<5<2olul`1Q^dvdwg#!VUspDL?vST~!^Y-R2D>c{z2@`9Jb)LAw3xal@XX4PEX)hVso{Zy#zKfkT53y7$L(Yu)v(SeWXSeavpv>0j#0FMHruNPF-;H~vDu(zreQAG}mn=={5CcQD?Q8aR~2VxrpuGga=@75rmCy>zJ&?<;6bye>yjBR~$drRGtYvtPd+RmM!8`^5rwM#_xb-D9nl^!q5hc{3EyZxixv5cHyXpa>^xZyr^;LiVOsoZ7>ZRW{%aVE6R_j%Vh-Zv=Z-`+UunOv+t93gew+}{>wlMBBt9(P4`qnSK+y$&q4tafwhm1<fzADOGCEQQ`}$aJ@sl|J`e|L_l#y7k(s7TfE0R-L=jboQ)r#=LIax`JnpcEU6FCU51j6&^_^^w<i|t4=cG7}y*X&C^jw%qeP*^ur{a{wQfM$y4}h*8$}I$3Md++PkJdc<bKYz1OkjzyHB^_x}C+_W0+Ue>_wwjwW@KlgZ;YUzF1j<1j;1t1~*8gsiDkrT%KP#;mi#(B#JMxF)zasKCs49oJ*D2a@&Yp?9j?QS-U?PB0gHOx}K*k#9}5r@4-=r)ve|&w1R2OKTNIF4ret?a<byZdW#N^cC*Hx{Vgf;w_(Zt{r8tR}Y7<=_WLd&deZmHpFmfi`gF%B*@)9N_SVQN4GZGJX^ayCtj0nYUZrYXSlMbs2!qmw#>56Wz{=u3+Gi4O!Q1d2jUUD=a_Y`7q>0jc8bUDY(;l1<gQbE=WL*5&F3(ihhonMZ+`RsAAf994mmzOJWB>FcrbI#8{Ux^Y?gWcPMPA61YX*@nZr5Lj!D4>Q*WPg?E*g(`C5eZ^H$cma9l0`*q1iE%i*ckXIHMcXV=F5w`)UwhYn_wATx6A*mVz$%{_5xhNE7GU-VzQjVlMkiJ!rwffeoC5H<ty#?4zhx-qfkt?g}BzqixB(l+42e%-PsfE!i9I#|iqVYW<%na}R@NA)YKO{)+u?$ZJnnU|~9(Q9k#>i{aj1XxyQ6wqi?>g#Kp7Ur^DHL?;KOBIF&`VZczaWXjT`NOJ+VHS3Dw!0)fE6rl0HPL|eohK%2gH00dt_S~K!Ztq9;Ef>VV6{8yrXD<AaL0i;?oTYnW9+uBZXXN~mOz&d6ZY}gLp}!LI=WFHCm#P{|2Gp{eKHXq;i#ID-hnAFH^OmxWyw{|b|K8yTnOW5F57WU?)u^Fe%3vE#h1svTXnMAW_<0%k2DFb!<Z9Y$DB+%PF|K+R;}yA<MZuuu5lYrUAvCu+Lb;I*RJ%bYuBgSwQIhY)#;<VcAb92zFl*=eXQ<X-D{AJ3_V;X<V7it56DRDbnKU2_-eoEP$-e=Ad1>utNp2o8VQ?LnI`Ilymgji_w;$)M5)x0yK`Jz9exi5WwqBob|bR0bu-#BZLnK)9r4mj_C09HHEGq$ooRGKZ|lK0s|~#)Z2b^duH_U=Nn4l7Y7k$5<=XDi0<s29v=4J)J?D6O{#8>g5}l8J6ET&J>xG;|d++~h@2%JOe(+Y4X?J0i)2v|hw%)Y4vg>Zl+sZff$Nqs}Gde3<+b=Dl^?IvAbY=-_G^+}NdoY)RtVzVv{?LoM#BMXPIWj$VY<}CHyuJU^cOSfUudS&weCY0A*K~ZbOTwc=-drrWomd<MQ-yPsw~Uz$%BvF<uhu_Txs$7m$Su~F2hd)3@CEE#sto2s;QbZgY8%!TF4s!^c+RRGkVLufqWIcHG821r{l-PA)86AM(els$&j!RcULH3i8_cQeP9H+^Wp?3vxudJmKfgAWkmLPif|eBL(!lbfZo-_}?RIKGFw^PiE52~pfp0d6nqBHktt_hMWJzwne}2ntUN+s?lW!MQ91K~r71`9^w{^F`%$!3K;90iNMwS^zJ<45f5@oK4sa>2=yEw~-M2Ds0XJ7u}ArTTL0y|sf0&_R&(Ux<8^Y`n^lX*#6W)gpO45vPu_<tKIH4cCcGQAbb9^D@K2QWT%Yu3i>)u38!Gy3)<^72rqWB{(pu2KbSosRx`?~T2;-`M}{Z+&i-dGS9pM22O(yL&)#EKpN7iQB>=aGjXdm$$V&{i%25cFwv!c{nsAn;$GNO?|lh)l=1DYrm{bEj?U4b**<<yxN=wbW4b>87BuS2+Pqkv)4c_jb!!0Ouoww=ur7o|BmWpT{n9fArFbUcqO{*{W`3Qw@Y$F5G|ryxq;Q@#KSvbxg(F@^rKDzR<n#Ff_9P*GX)#|*{YT!Ha`LWymeE#tC+)cXuwh(+23+YJ*HVM8crTu|9<E-2+wK)$AG5+8&3l^9(TY7c40TR)bu<^cWnb|s?HgFoXe3ka^5B`mf?7pBQ%Pci*ZVtBEg5f9+`Xl@C4i!xUw1BFnsQ;`m%GQP9pI3th<{?w916%lE{Rm7RZEGZ_Jbv;2|Fc`S5)69>6PW<4NyW-5THoZ(8}{{QA2v#&(3)#(o&dc&6ZjZB7?8cNMs609hBnnh9WBXooWy$~=5!%?d8sx$^VBGzgzFh9J448-{z<<SuQc=+3sdt$nezi@nVUxT^;hrp38gxVDf*FY)weh8oks#<ZWnd-;3}fx96#Z-^Z*ZIkc2Uy9CNe6)i{m(A>Yccg8T5NY;!%ptBrr1jH=3h6aK_8;SG9kBJQ7U+VDu<-iJbBB)<xVpLU0<6CMSq<)Yn!Q2q(y4RdG<@0AFy8=!PCI$PIu}PLcn4KqIt^S1RhL0~AH28svwOZqkgwJSJ=mlp8<6VLogMRpO7zob&ip+HbvKQta{Z}XKM}c3SCXfaeX(S3iq`oLSI*J##F6=<5&0)5-{+3}1O@)rcXrmE3j9w*;D7y=zNI@f<4RCW50d&p4(KK^vTyjQ(F)_L7TEbwypo)y@$#|N#JDOMc76mAc(zkj?}<<w<LCujv0RByv;KIYEzQtVAZP38b)3lB7sf3j0wBwWS4;RDH2mb1)J(~kuv*NbN-;;N9PscJVjf|2n4?vO8K?^LY0&R|FG1^7GpzHErO%tP9S3Rjj{Kih6m>7l$I$(|<5SZcB2Lm%le?#0dEwdTFTJp`p`iGwHZo@>%a~EAiIwE0NLgZJOpM@y@;ELGQABE<)2yULDtJOuVUk?tx#X}ad@ylS1Kk@Gnw3Oy9v8WhR8n3DQyMPwqDW0y#5~S~=2Y{P7iGb?6ud}^f|gmH#ATi&EnIzjXKVctLwEo0L6>nBV{U0lr7UAZQzKbnjFefXsSsHvbE%Y04M*2JPLn(@39y&YT(OwOgP`m5)xHa%;|za9k}H~!qLdh2DN{u<!)d08Ny{W9rO}1bC1FWksvLtZg~Icc=EGfV>+6u;+jpvIWKv`*)&z^L(zGZTD`_q=DGMe_t))s-ri>7*<ax&PG&81TEKezmNvX|nAJdy!diX`gOIBcnlBA-fDe$6;OqHT!nn@*SA$3Zm%+o}cG|2?h!Z0jSMzl~`^x@aFGJF{h*9EY{f!LJkgcXS>6RALVROT_oUl60l$2bM`TFkMWXpBRYLMY6NQ`UB|A#Of00<y51tnGvtrE{GV8tW2tQ3#rtn5Y;?q6#dOmy%`%_$UR&pX0BRc^M0k7ti9N1?vU(BO7%EzA9wrDMkpOkSEeW2!QZ(ZU`e~YQW}+E`*S==DgIj<RFohNvTW%Ho$l|$<uNU@Gj{qnUy6_g!Y(<(Xud^(EyzTZmB5Yj1*~{sjTF2o@InEj4GCz#AQ)tr0t}3eDZL!3{p!gMR>+j6_=U|37QicL?g=t%BECYC5fU8M8y?REHCh0VbE`80&=Vf@sVM^hF|^b!9&tyxk{;#DWh@1Ssv?xNXiU&1=OOoPV)@IFR50T74V{q6yO!lB~#g4@IAppV@aqXh9n7*j48tuOBpzZWHM8QNDcmGjOHdwnWapi9deLk&AHJe?4~jf9wDCU9@=sMUh<ThiK0wJET~Zy1`!DqmP)bcN|ad|3-rka7)qxM(hhiKNyfEFWfrrAU3Bq9IEgcp6&V<S5Lp@~h}Mw7!r-F;yJT_7VpFC$=Bjx~R4%i$i1R#86(ftgYK1AZ36mgolJXP+lc==FiZYflmPqn6VJuTAH?dSK;nWf~&6ze>C9W6<2mN@1;ru^Z>)~1GeEm;#KQFsqDmKY;Q$V)?<%I?#&=Q0|N=`6E@I9w_k(m@!Wm2%T0^4KNOb!$$9Nh&S)yp28*t%DM?Owbxe=0Q~0XdCLfe%7s^)YKKeku)D83ravj%DYxfM$?9kD;8ol^Z-KrA*QQv~~3MfsJi(hqfP17%;0+NeL+cC`mIKGpNO!<xr+Fkr+hEWnz+&#VpNI$uh#V%DFLVs@s0N2`B6-%;^ha2P%>mX;yGTQ~WDgoQOi_GDd$=mntcX1ZXocO$qcSrbhCTB|6J#QnY<d*KCB&YG^f9T@<OxB+LK`rjHH5tYAi1u_5DxS}n{NOn{`!DTuZ#a_DIWvcnislDRItr|=_K>I!hz1-W8523%{ZV;DOyAgoj@80?{(a+0K2>$rsAiu0t@um~Uu2-u!N!zKdrV{r?8z&E<;f*J@op2E5+cnYbYWd`D=1+*KO7G^LEGy#dLiH<=O31lRNd6AYG(<C=-7dvgWh}EXzgzqG^WTR;exta+GSeEBdZCq#=dIc<LYbV4MUuc?_0&8U18yJM`O><~FbmHQ5wkxw#YK$?5s--MV62W1#QD7CrB!CZD!C`w4f%f1P0F*#JL?TF*VX<SYAzYrTRqITBzR9REU1fyW%olbmU<6>vLKaZ|5O*o$0#R8iKylb3K$0rJ&SKcbTGF`WDb*lN$uMfDi@52cgY5#AtjY+@CDzJHWS(nq56{5DOmdchxKja3)IfTJk_5~Hn4!5<-I5kr>l1VGbXWm~jP7{#OBlrpl2C9717A~Xg+u-%G3N!05D;GuGop|N7%M3$cwt1&vxMP02Yt>h{snjjb`|V>nxQ$=h%Pe<36sS@gcP}uST0y3nkyhZ2hSR47tX+%FyramYC$H(q+xUt4eP03z^x=8Fu|FD2-9F5bi!d%fFLqPM8+hArY9+W$)!o=6_&B|#j#^xl!!tQ5-SF?Llgx_n1Zk&W^<@lp$RWzmFW!K8kk%L5-|a`56&=dp!)3Efea}&mL&&oDsVXiy^FFW;F?nA3W^>ZFRl!U1w<9}Qcg>j#_^)`yPz-#b?jpyIjpsm<O!rop#|W>f-p%zgtf2|k?8{TOrS?-!j*vF7pda&GL<HUwOMJJ#Go-ok^;uN#6A&HJ%J^aX{M|~%}OW~m{79B=DCbDwkk2~Tba+xc?La;^<c1FNM7g+76fF2P6#AX41o<BhC&QrSA|yyNul~IMHdj25G$oKH->Y|P|L-Nfc>x)D}i2Y2#p4`L>p`=!Huv2v3J7`fR?Z-SZcRkGS{gRpz6Vh=+Xkg{ur}QVUK2t8Vw?ah{L8pL483j=FrQK4q2+PXxR2i%bsbTgHy}7af!JuSggPb4D1F&NI+|0i8yxOB@hk+Hs{!ZV5^NaC=1AI=xb0N7)4TufW;9z*tHOFV5Va$%&>KYU@#fb3%wh21u=zP2G51CdmZdMLDhy83Nt~sG*g2vdJ5D^8Fn0o#Sj&+KVgIFSW{;?q@|KrK)VkyVB(kosTgSv+bgyDO~nU#SZPvP;}7(jLALQUg>3>|4&8)5sj@tSEpK5I_B5bHZjTDe68i>d32b!a!pnXFen?<6aY0}fW}@IUw(3cO<tfZiW9hd5eZb;@x>7Ksiv;$gVc2bBU%2pM3A%)-gpv$G!W39V4pCuXY-kES%uDnL(I-;qJ!m}!8!uOQpkTN|A13nD?(W|JbY7aA"
SOURCE_PAYLOAD_SHA256 = "65564ca49bef1a8325c03b359b501602a798d5d4d84a36605eb860ddef2918f2"
MATCHED_60M_SOURCE_SHA256 = "0238a5e39c68d0625decc976abd7b27e15216293179e1132d5fa2ee5c1f814ee"
VERSION = "SOURCE_K234_AUTHOR_RULES_INTERFACE_ONLY_20261019_V1"
SCRIPT = Path(__file__).resolve()
CANDIDATES = {2: (0, 1), 3: (0, 1, 3), 4: (0, 1, 3, 4)}
NEW_COORDS = {3: (0, 10), 4: (10, 0)}
METHOD_FILES = {"SPF": "methods/minimum_distance.py", "STTF": "methods/minimum_sttf.py", "QTTI": "methods/minimum_qtti.py"}
DEFAULT_SEEDS = (123, 124)
EVAL_MAX_TIME = 600


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(path, rows):
    rows = [clean(r) for r in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    os.replace(tmp, path)


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(values) if values else None


def sd(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def snapshot():
    data = zlib.decompress(base64.b85decode(SOURCE_PAYLOAD))
    if hashlib.sha256(data).hexdigest() != SOURCE_PAYLOAD_SHA256:
        raise RuntimeError("内嵌源码快照校验失败，拒绝继续")
    result = json.loads(data.decode("utf-8"))
    for name, content in result["files"].items():
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != result["sha256"][name]:
            raise RuntimeError("作者源码内容校验失败：" + name)
    return result


def materialize_source(root):
    data = snapshot()
    target = Path(root).resolve() / "_author_source" / data["commit"]
    for name, content in data["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("内嵌源码路径异常")
        destination = target / relative
        if target not in destination.resolve().parents:
            raise RuntimeError("源码释放路径超出本轮输出目录")
        if destination.exists():
            if sha256_file(destination) != data["sha256"][name]:
                raise RuntimeError("隔离源码已变化，不静默覆盖：" + str(destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content.encode("utf-8"))
    # 只补空包标记；保留原有含空格的初始化文件，不更改源函数。
    for name in ("utilss", "rl_env", "policy", "methods"):
        marker = target / name / "__init__.py"
        if not marker.exists():
            marker.write_bytes(b"")
    return target


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
        self.text = snapshot()["files"][self.path]
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


def setup_source(root, k):
    """每个条件独立子进程；只在内存里复用60M的增站配置。"""
    data = snapshot()
    target = Path(root).resolve() / "_author_source" / data["commit"]
    sys.path.insert(0, str(target))
    from at_obj.map.map import Map
    if k > 2:
        original = Map.__init__
        def extended_map(self, *args, **kwargs):
            original(self, *args, **kwargs)
            for vid in CANDIDATES[k][2:]:
                self.vertiport_station[vid] = list(NEW_COORDS[vid])
            self.vertiport_station_num = len(self.vertiport_station)
        Map.__init__ = extended_map
    from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
    for vid in CANDIDATES[k][2:]:
        VERTIPORT_CHARGE_RATE[str(vid)] = VERTIPORT_CHARGE_RATE["1"]
    from at_obj.scenario import Scenario
    from utilss.uam_rl_wrapper import UAMRLWrapper
    if target not in Path(inspect.getfile(Scenario)).resolve().parents:
        raise RuntimeError("误导入项目已改造环境，而非本轮锁定作者环境")
    return Scenario, UAMRLWrapper, target


def passenger_metrics(scenario, counts, restricted_area, reward_sum):
    persons = scenario.persons.persons
    finished = set(scenario.finished_ids).intersection(persons)
    passenger_rows = []
    for pid, person in persons.items():
        records = [r for r in scenario.person_travel_records.get(pid, []) if r.get("end_time") is not None]
        complete = pid in finished and bool(records)
        last = records[-1] if complete else None
        passenger_rows.append(dict(passenger=pid,spawn_time=person.spawn_time,decision_time=person.decision_time,
            departure=person.origin_vertiport_id,finished=complete,
            finish_time=last["end_time"] if complete else None,
            ATT=float(last["end_time"])-float(person.spawn_time) if complete else None,
            source_ATT=float(last["end_time"])-float(last["start_time"]) if complete else None,
            source_AGT=person.time_stats.get("to_vertiport", 0),source_AWT=person.time_stats.get("wait_uam", 0),
            source_AFT=person.time_stats.get("fly", 0),state=person.state,sub_state=person.sub_state))
    completed = [r for r in passenger_rows if r["finished"]]
    full = bool(persons) and len(completed) == len(persons)
    denominator = max(1, sum(counts))
    shares = [count / denominator for count in counts]
    entropy = -sum(p * math.log(p) for p in shares if p > 0)
    result = dict(generated=len(persons),finished=len(completed),full=full,completion=len(completed)/max(1,len(persons)),
        ATT=mean([r["ATT"] for r in completed]) if full else None,
        AWT=mean([r["source_AWT"] for r in completed]) if full else None,
        AGT=mean([r["source_AGT"] for r in completed]) if full else None,
        AFT=mean([r["source_AFT"] for r in completed]) if full else None,
        ATT_completed_only=mean([r["ATT"] for r in completed]),
        source_ATT_completed_only=mean([r["source_ATT"] for r in completed]),
        AGT_completed_only=mean([r["source_AGT"] for r in completed]),
        AWT_completed_only=mean([r["source_AWT"] for r in completed]),
        AFT_completed_only=mean([r["source_AFT"] for r in completed]),
        restricted_person_time_per_generated=restricted_area/max(1,len(persons)),
        raw_reward_sum=reward_sum,simulation_minutes=scenario.time,
        action_counts=counts,action_shares=shares,effective_airports=math.exp(entropy),dominant=max(shares),
        aircraft_ever_created=len(scenario.evtols.evtols),
        metric_note="原源码未建模最后一公里，不额外补算；源时间分量原样统计，不强制修正分解恒等式；未全完成ATT为NA")
    return result, passenger_rows


def run_episode(root, task, seed, author, classes):
    import numpy as np
    Scenario, UAMRLWrapper, native = classes
    random.seed(seed)
    np.random.seed(seed)
    # 参数及顺序与60M回放一致，不改满座、自动补机、结束或奖励规则。
    scenario = Scenario(max_time=EVAL_MAX_TIME,person_spawn_file=str(native/"train_data/passengers_300.csv"),enable_logger=False)
    for vid in CANDIDATES[task["k"]][2:]:
        scenario.vertiports.vertiport_evtol_capacity[str(vid)] = scenario.vertiports.vertiport_evtol_capacity["1"]
    env = UAMRLWrapper(scenario,list(CANDIDATES[task["k"]]),2,num_frames=6)
    env.reset(seed=seed)
    counts = [0] * task["k"]
    routing = []
    restricted_area, reward_sum = 0.0, 0.0
    started = time.perf_counter()
    try:
        while True:
            action, detail = adapter_decision(scenario,task["k"],author,np)
            if detail is not None:
                counts[action] += 1
                routing.append(detail)
            finished = set(scenario.finished_ids)
            restricted_area += sum(pid not in finished for pid in scenario.persons.persons)
            _, reward, terminated, truncated, _ = env.step(action)
            reward_sum += float(reward)
            if terminated or truncated:
                break
            if scenario.time > EVAL_MAX_TIME + 2:
                raise RuntimeError("作者结束条件未生效；只标记失败，不改环境时间设置")
        result, passengers = passenger_metrics(scenario,counts,restricted_area,reward_sum)
        result.update(k=task["k"],method=task["method"],eval_seed=seed,wall_seconds=time.perf_counter()-started,
            condition="SOURCE_K2" if task["k"] == 2 else "SOURCE_EXTENDED_CLONE_V1",
            original_k2_parity_checks=len(routing),expanded_decisions=sum(x["expanded_choice"] for x in routing),
            evaluation_trace_role="与60M相同作者训练trace回放，不是独立训练seed或最终测试")
        return result, passengers, routing
    finally:
        env.close()


def worker(root, task, config):
    directory = Path(root)/"cells"/task["id"]
    work = directory/"native_work"
    work.mkdir(parents=True,exist_ok=True)
    os.chdir(work)
    logging.disable(logging.CRITICAL)
    from loguru import logger
    logger.remove()
    classes = setup_source(root,task["k"])
    author = AuthorFunctions(task["method"])
    save_json(directory/"AUTHOR_FUNCTIONS.json",author.manifest())
    started = time.perf_counter()
    rows = []
    for seed in config["eval_seeds"]:
        output = directory/"episodes"/("seed"+str(seed))
        existing = read_json(output/"SUMMARY.json") if config["resume"] else None
        if existing is None:
            result, passengers, decisions = run_episode(root,task,seed,author,classes)
            # 先写逐乘客和逐动作数据，最后写完成标记，断电后不会跳过半成品。
            save_csv(output/"passengers.csv",passengers)
            save_csv(output/"routing.csv",decisions)
            save_json(output/"SUMMARY.json",result)
        else:
            result = existing
        rows.append(result)
        save_json(directory/"LIVE.json",dict(phase="rules",pid=os.getpid(),finished_episodes=len(rows),
            total_episodes=len(config["eval_seeds"]),eval_seed=seed,time=now()))
        print(f"[EPISODE] {task['id']} seed={seed} ATT={result['ATT']} completed_only={result['ATT_completed_only']} completion={result['completion']:.2%} actions={result['action_counts']}",flush=True)
    full = bool(rows) and all(r["full"] for r in rows)
    result = dict(task=task,episodes=len(rows),full=full,
        ATT=mean([r["ATT"] for r in rows]) if full else None,
        AWT=mean([r["AWT"] for r in rows]) if full else None,
        AGT=mean([r["AGT"] for r in rows]) if full else None,
        AFT=mean([r["AFT"] for r in rows]) if full else None,
        ATT_completed_only=mean([r["ATT_completed_only"] for r in rows]),
        source_ATT_completed_only=mean([r["source_ATT_completed_only"] for r in rows]),
        completion=mean([r["completion"] for r in rows]),worst_completion=min(r["completion"] for r in rows),
        replay_ATT_sd=sd([r["ATT"] for r in rows]) if full else None,
        action_counts=[sum(r["action_counts"][i] for r in rows) for i in range(task["k"])],
        effective_airports=mean([r["effective_airports"] for r in rows]),
        wall_seconds=time.perf_counter()-started,status="COMPLETE" if full else "EVALUATED_WITH_CENSORING",
        seed_note="仅环境回放seed，不是训练seed；规则无训练学习曲线",
        fidelity="原函数原参数；新增站的外部循环沿用V1配置",time=now())
    save_json(directory/"CELL_SUMMARY.json",result)
    save_json(directory/"CELL_END.json",dict(status=result["status"],time=now()))


def collect(root, tasks):
    rows, episodes, failed = [], [], []
    for task in tasks:
        directory = Path(root)/"cells"/task["id"]
        result = read_json(directory/"CELL_SUMMARY.json")
        if result is not None:
            rows.append(dict(k=task["k"],method=task["method"],**{k:v for k,v in result.items() if k != "task"}))
        failure = read_json(directory/"FAILED.json")
        if failure is not None and not (directory/"CELL_END.json").exists():
            failed.append(dict(task=task,**failure))
        for path in sorted((directory/"episodes").glob("seed*/SUMMARY.json")):
            episodes.append(read_json(path))
    save_csv(Path(root)/"BASELINE_RESULTS.csv",rows)
    save_csv(Path(root)/"EPISODE_RESULTS.csv",episodes)
    save_json(Path(root)/"FAILURES.json",failed)
    envelope = []
    for k in sorted({t["k"] for t in tasks}):
        selected = [r for r in rows if r["k"] == k]
        valid = [r for r in selected if r["full"] and r["ATT"] is not None]
        best = min(valid,key=lambda r:r["ATT"]) if valid else None
        envelope.append(dict(k=k,evaluated=len(selected),expected=3,full_methods=len(valid),
            best_full_method=best["method"] if best else None,best_full_ATT=best["ATT"] if best else None,
            note="只在全部回放均完整完成的规则中比较；不是数学最优或下界"))
    save_csv(Path(root)/"FULL_COMPLETION_RULE_ENVELOPE.csv",envelope)
    progress = dict(total_cells=len(tasks),evaluated_cells=len(rows),evaluated_episodes=len(episodes),
        full_completion_cells=sum(r["full"] for r in rows),censored_cells=sum(not r["full"] for r in rows),
        failed_cells=len(failed),time=now())
    save_json(Path(root)/"progress.json",progress)
    return progress


def launch(root, task, config):
    directory = Path(root)/"cells"/task["id"]
    directory.mkdir(parents=True,exist_ok=True)
    if config["resume"] and (directory/"CELL_END.json").exists():
        return
    if (directory/"FAILED.json").exists() and not config["retry_failed"]:
        print("[SKIP_FAILED]",task["id"],"如需重试使用--retry-failed",flush=True)
        return
    save_json(directory/"TASK.json",dict(task=task,config=config))
    save_json(Path(root)/"RUNNING.json",dict(task=task,directory=str(directory),time=now()))
    print("[RUN]",task["id"],now(),flush=True)
    command = [sys.executable,"-u",str(SCRIPT),"--worker",str(directory/"TASK.json"),"--root",str(root)]
    with (directory/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        options = dict(stdout=stream,stderr=subprocess.STDOUT,cwd=str(directory))
        if os.name == "posix":
            options["start_new_session"] = True
        process = subprocess.Popen(command,**options)
        save_json(directory/"LIVE.json",dict(pid=process.pid,phase="starting",time=now()))
        try:
            code = process.wait()
        except KeyboardInterrupt:
            # 只停止当前脚本刚创建的明确子进程，不使用pkill或killall。
            if os.name == "posix":
                os.killpg(process.pid,signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=30)
            raise
    if code != 0 or not (directory/"CELL_END.json").exists():
        save_json(directory/"FAILED.json",dict(exit_code=code,time=now(),log=str(directory/"run.log")))
        print("[FAILED]",task["id"],"后续条件继续，详见该条件run.log",flush=True)
    else:
        result = read_json(directory/"CELL_SUMMARY.json")
        print(f"[DONE] {task['id']} ATT={result['ATT']} completed_only={result['ATT_completed_only']} completion={result['completion']:.2%} actions={result['action_counts']}",flush=True)


def archive(root):
    root = Path(root).resolve()
    destination = root.parent.parent/"experiment_archives"/(root.name+"_DATAONLY.tar.gz")
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp = destination.with_name(destination.name+".tmp")
    with tarfile.open(tmp,"w:gz") as output:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in ("_author_source","native_work","__pycache__") for part in path.parts):
                continue
            if path.suffix.lower() in (".json",".csv",".txt",".log",".md"):
                output.add(path,arcname=str(Path(root.name)/path.relative_to(root)),recursive=False)
        output.add(SCRIPT,arcname=str(Path(root.name)/SCRIPT.name),recursive=False)
    os.replace(tmp,destination)
    print("[DATAONLY]",destination,flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=None)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--ks",nargs="+",type=int,choices=(2,3,4),default=[2,3,4])
    parser.add_argument("--methods",nargs="+",choices=tuple(METHOD_FILES),default=list(METHOD_FILES))
    parser.add_argument("--eval-seeds",nargs="+",type=int,default=list(DEFAULT_SEEDS))
    parser.add_argument("--phase",choices=("all","run","pack"),default="all")
    parser.add_argument("--worker",default=None)
    parser.add_argument("--list",action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else SCRIPT.parent/"serial_runs"/"uam_source_K234_SPF_STTF_QTTI_v1"
    if args.worker:
        record = read_json(args.worker)
        try:
            worker(root,record["task"],record["config"])
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        return
    if args.phase == "pack":
        if not (root/"SUITE_PLAN.json").exists():
            raise FileNotFoundError("指定目录没有本实验的SUITE_PLAN.json")
        archive(root)
        return
    tasks = [dict(id=f"K{k}__{method}",k=k,method=method) for k in (2,3,4) for method in METHOD_FILES]
    selected = [t for t in tasks if t["k"] in args.ks and t["method"] in args.methods]
    seeds = list(dict.fromkeys(args.eval_seeds))
    if not seeds:
        raise ValueError("至少需要一个回放seed")
    if args.list:
        print(json.dumps(dict(cells=selected,eval_seeds=seeds,episodes=len(selected)*len(seeds),training_steps=0),ensure_ascii=False,indent=2))
        return
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("结果目录非空，请使用--resume或新的--root；不覆盖历史结果")
    root.mkdir(parents=True,exist_ok=True)
    data = snapshot()
    contract = dict(version=VERSION,source_sha256=SOURCE_PAYLOAD_SHA256,eval_seeds=seeds,max_time=EVAL_MAX_TIME,
        candidates=CANDIDATES,new_coordinates=NEW_COORDS,new_station_profile="V1",wait_interface="作者get_wait_time原函数",
        physical_environment="与60M锁定源码逐文件一致；新增站沿用同一配置")
    signature = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    previous = read_json(root/"SUITE_PLAN.json")
    if previous and (previous["signature"] != signature or previous["script_sha256"] != sha256_file(SCRIPT)):
        raise RuntimeError("脚本或实验配置变化，拒绝混入旧结果目录；请指定新的--root")
    materialize_source(root)
    plan = dict(contract=contract,signature=signature,script_sha256=sha256_file(SCRIPT),tasks=tasks,
        source_repository=data["repository"],source_commit=data["commit"],source_files=data["sha256"],
        matched_60m_source_sha256=MATCHED_60M_SOURCE_SHA256,
        function_contracts=[AuthorFunctions(name).manifest() for name in METHOD_FILES],
        adapter_notes=["作者文件完整保留，仅抽取原函数跳过失配顶层入口", "旧四乘客观测转当前焦点乘客，一分钟只执行一次原环境动作",
            "旧v字段映射到已揭示的enroute/to_vertiport，不读取未生成乘客", "SPF原代码仍含等待项，不偷偷改成纯最短路",
            "保留SPF除数2、STTF/QTTI除数1.5，保留QTTI的30/6分母", "K3/K4新增站只在适配脚本外层复制V1规则，不冒称原论文已有扩站"],
        metric_boundary="与60M同一作者训练trace、600上限、原满座尾部不修复；不完整全体ATT/AWT为NA",created=now())
    if not previous:
        save_json(root/"SUITE_PLAN.json",plan)
    save_json(root/"SESSION_START.json",dict(time=now(),pid=os.getpid(),selected=len(selected),episodes=len(selected)*len(seeds)))
    config = dict(eval_seeds=seeds,resume=args.resume,retry_failed=args.retry_failed)
    print(f"9个条件；本次选择{len(selected)}个、{len(selected)*len(seeds)}次回放；CPU规则，无训练；作者commit={data['commit']}",flush=True)
    collect(root,tasks)
    try:
        for task in selected:
            launch(root,task,config)
            collect(root,tasks)
        progress = collect(root,tasks)
        save_json(root/"SESSION_END.json",dict(time=now(),progress=progress,status="FINISHED_WITH_FAILURES" if progress["failed_cells"] else "FINISHED"))
        archive(root)
        print("[FINISH]",progress,flush=True)
    except KeyboardInterrupt:
        collect(root,tasks)
        save_json(root/"INTERRUPTED.json",dict(time=now(),note="使用同一命令--resume继续，不重跑已完成episode"))
        raise SystemExit(130)


if __name__ == "__main__":
    main()
