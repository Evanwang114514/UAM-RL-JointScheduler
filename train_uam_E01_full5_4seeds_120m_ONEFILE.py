#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E0/严格E1完整五组合：120run，名义120M，实际120,422,400步。

中文实验说明：两种环境、K234、五个方法—控制组合、四个独立训练种子。
作者源码及已验证严格E1以只读快照内嵌；不依赖其他矩阵入口，不修改项目env。
仅新增表示与条件联合策略；不改乘客、坐标、奖励、充电、满载或时钟语义。
训练全部完成后回放；单格失败后继续；--resume恢复；--phase pack仅打数据包。
科学边界：E0没有空飞控制，三个J3组在E0关闭飞机策略，不称真实联合调度。
"""
from __future__ import annotations

import ast
import base64
import copy
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import textwrap
import time
import traceback
import zlib
from types import MethodType, SimpleNamespace

# 此快照只包含既有严格E1入口的定义，构建时删除了其__main__执行段。
# 快照内部作者文件逐字保留；本文件不会调用旧矩阵调度器。
_SEALED_BASE = 'c-oA*TNnD;aVPlhPjN6p3sBS%64youbU_z{Ku7`s4+tb~5|R*-5aPO=PP-GQ<HXaG*h!qkNhg_H*2GCqRubnp9zW!x^!fkK@l$_=;n?H!OwT+t4_dX0tzCOp)vo&e)>1a}i{~i(Mbq|wapNbS1lM?f^uLn*1TMm#SjV;P@BZy4=XeX(ztnu?t6%-<XFvF}kAL*fAHV&#-O_*i=imGAn}7A;Z~wz*Kl=SIzV`Jme&w$|{Mui9@yCDm@%KOd_}w3W{N{iD><6EI_|~s~_Jgne<WK(at^E8iKmDojwV(X<kALzz-}~^Z-}&OlzxCm}fB0X&_V2#(l@H(i?T_F7mCt_o4<G)+*FXIIum5bxkH7h?&p!Rr55M_qAO7+G`1t$Z=prA!{?{MA`OBaE;8#BXy?_7Nr@!^_+yC_O?|%D>@Bj7ZKm6s7fAoV7KmO(a_Rnv7-~8<V{DUw4{@WkE{tuu1@Y|n%?=OGyJAeP->wos)H~-{IOP~Gt4?h3x-~I3#-z$Ik^v^&3@Q=Uv(U1P?*M8~4*T4SxZ~gt}zws+?`6pld-iJ^Bx3}=cFaPL^uYKpsfB)>;AAaq>yoJyH>32SS|3ANlkKg%+pZurqefERj`22_e>&HL%-p4=q<M-CTTweatEh75io4@q=pMUeicR&5`{oi{Z{e9ej`Oe$s=im6%G092vd+^=wMw`w0U%pe4-h2M=t#5wzPv0mb(oc8z!>@nw=YR6?55M){d!K&(>5qT%wQqd<kAM00;%7hjtHFD&h+^drU;m4r{PJIX_M;zv{#$?X@vr~%t^Tfzz72o&gKvoc@|}P6%umg{WA*>BRQ;Ky_c9;;>^mg{3D5qA@BiUXe)sEd`4_MX`U?nxd=iH%KeA){=tuAMKYshKKmT|C^zoPe^M^nE=9f<Tt>69PZ~o?sfBd^Ie*C4+einld-+KGw?Uk>6@i+hevw!~EpC0BNjURvi<8S@bPyYCyKL76b-%0kf5Pk9EKm7O`?^OBbhhO{S55M~tKU04A{x`n(ldu0QdT*0o9`0BF@ts#+{PC}U{KlWZpP&8U*FO8_KYF|UiywdMJ^b*${OGeE{^^%y-kJH8ZFuxPF^ugwK8G8__{8>tV;Fz3^8I7H`sp#g@|B<K!d<Wmqs`C9<|^7$%Re7kQSx(fzl!#*ZT@^{`%$p5;-3$l=zUrKZ`nQWcJJ%*^P#tj_dl19(a-i4ymjxze!emJ|1st5>h;n4`6#k?{>uH=(U-fAV%v)TRdY6j@MvwK|E{_Is~E4st?mBn(PG>C?{c`ZHs;DY{9JMS@@((x{!Y1{{O-3t{OUh^{Qdv^iywdY^S}6yAO6#?e|e$*@xOfd)$e}z;ivEW_TjI;^|Qa;Ir)D6EMo7#e*Skqd<XsGuYUV|**^R9)1Q3(o1g#a`(MUndkjA@{FS$PyX}+wSBH{p491e7@k*Cu)UW=+CtsDs{{N^N%H09gC)NK^{~mUyvhx2@7yj*6K6(F+(Z6}8l^q`ae-mxu^=7*|yK(-fmftUyzFPRo|No4y$}^204#op&%(7jj%Y5~JYP!bO|EDGj!|s3{GQxoSa{5n;==+tgsKc>BcMX}G^@k+$N$ZoZT5xDT*NI{74~fE$=$$jT9>zHBVwL24T4j%lcy^h=(VaxC842uViQO>$E5fv_vUEKW``i?67KPn~u)ID;y|gKvm6{#aE%x97$X6!K{Pf<h<HMDJzS2VNzU?Ap{;;`WC10SZ%-9J#!^0tJQ9Yg5K6}Yrp63ILX_>Rb2=^P8=4ewl#-NcmgxkKP77tWQv?rH7IBI)Ak0o`vxtzT=H+c(L;swvz-F|IsPh$HjTwiswpTBi^bOujB5i}3!)LaES^Qd~|fs9<nyJ$ZM+_f-&SW5F$_Q=(;1$5LkpD`}-g>-Y<%B~;l%~eq@*^+-7j&}{ll3$(b!fxD`S0$&F7m<CPUaMH@0{48ET%Nf5@P2q*a+?mD+P9WBEk@HWh@nP0Zxj4=euC5u&2BIJ80pb9PgxI+izQK%_~RsR(?*M?kW-1t_i;%9YNk<~cqmM=S0P-~8?*yAqlP*>L&fF>Yz?ZeF<6WUs@PIQChPZG&ovq{B-~<~)`@=(z<MWUmQEMT=NrnX1+Rc$I@Nf(dtn~c@u*n91ey&<c-Bm(1^E(q`0>yS2>oc4$Ak93p9YcMV(c0%T0N6%!53kfwnJ0j99Cg*SKgW{Y&4j2sHWV6fhHsnT3w%dxd)SR+2y_|&unFRz4;~Q(C|9N{7M5CN)_KtGPU7QQxM_NoQ=lmAv;(PFBb0X-JGoEXTg$*hEw&db*quSAJpt?+nd>HA5o=Vxs~$@tPAJQ_vBvRH)f7Gn@-gGV0`2|Wpd8f?ebM{SS52S%G<U~LWP7rh!^H*0mEMPVantB{;X78|B~}?YnID7{q|&(JahF9fktwT$iHWt;h_Vfn`*l|g{V-^dts>|k2FW-pRB2{9CqZl{zxeGFteLF`wjH=SMAyL{qbtd=ex5zvA{-;`eA)nMQQLTk?A0rj7sC`{V-u$gGShRGz_7j)*cHnQ^l9c@!(Fx#&KeV>vUhMwS)sqJ3D8Lh&gAxMmCf>iX>P==V7xRrG(^aB7!^sHYsM-Vd5>bCTi1Er!Z(qRl&vNLJ3G<b1qZs10QB~aGa-H_njWp3*+wD*DcAv+RNc+cq`jj5z+J}c~pd*Y7F(BPo1|^E^l@tksQ(b%2d_x)|@>R@e-f!1aPBJLSmF!wlNg}qfDjzrBTl8KrOcf$XsnWZOC}dy=vt~7ih#SX@j04dBM%{msJT_d26UsE~H*_Rl3?(n@_jOs#p@yUazps*zJC;Wnd&V-S!vmzK%8Yk@dK=y$r2R0iTvsHBY$ML|z1LQZBv}6u$e#l7#dgSA*d2ZddSUfwXT8wij}uOMNcOPnx{FSDK+6COIc)564_4JbT_~f1YL5`wWr_fD+(dKT}h}Wvy9+rhMi+TzAMb1F8>&ze9mgeHM<uykXYbC-YW$*1HuTp*t<HSvk17@@}*1@FhZjz#KmDTDR@ou8J36+0Dh>b7>=2bntZ=wS^!-^M`#V1v+SWU<b43a@grA*VUuC8B}u0YKCTUh>1!Rcub0OExXajnQiJI2Q+H-2kChab5zsozp~342YJj@+1sO#sKTf94LR7)qf*Gl!|Wd4DPax7hZF;o(??cvUAERf3;j-`sowgVak!a-QgAl-u16zYM6b?Qrg`gL7QsQ0>xR8xdu*Yt_#oY`w$>)yM?-sZorZ2wpgWhTG;wLUHSmyKHf_U>!FQ(i;|#xm>R?V?m+>iDY=rzvqKdbAV%U{+({so1OaP%BKa%sje5l-@vbLLYG=J(E2EtvC?dkrUSvl0ZMV;eXyG{4QkY*Sag>KLLG&R?`nU)Ann4L`TobQ>#$x#mht5U-qd)~~nn$9visJ8PNH8Y3qDA9STm&uWL=BKlw_NjL6Y%+>oS!MP>qjt%)OV}FeV?BP;9QHPm<PL+|u64mu5Vcp7ovjt)cB~0zZtiN-d3MX{m3GbtwIT7E_~WxOH0|5#Q675sY7up4eB17<&)K-H4%U~AV%G}iHnThwbJ4BA6|4z}P-xaID#Kp5OAn>1+s@Ypmt{&}C<h@H*Iwn@uL<05I;~k?6~s9N^9xrZ*K}pjhjOK2_e5F=e|=@NCJ0@DJ=?7%Rh&B`)vY8H@+E1}+n0pablP6ar<uLWn%V`674_gcC&Nz0xyJdjJO`;0C_aZDw$z~bsGsvyvhna}2Se1wwC3KdP3an&@ipPC7MDAk!}#12;%v7*U0wi0z6{yj7!L*-B$z;a2x(wkXY8Z?$dOl>?M*9Yt~v4IX{R{lmkE=2g^I|*^krK-Oo33@vg*(zIEEL?$9ZXH_$9$IRdC?uHe4Q|(W7i<FKFw5R0mbYqPImUP5c*cn9#Mn#G!Ik2G8(SP-G6vNO^2%2!41IIvvg}+d0nRhASQ~zF0bAFI$tzBcyF<*ozr8c(uKKt<Lb8x|>|H>)7j-@Cm8y0|d^E>k7y(VH}#~GTjJ7iaZT8+n_b6bVuNNlJcdqF4&;ED{4vsSI%?Z*dG?Gm3MY}2Tlit%dtDw(oh}&-IZBMYHI;Gue&Zd%tYQ=vpNz+wQCXWCHFnczmg{GoySHamv+b+0>>g(X`0S$kP#IfjM*!f;h%Xl6ZaHt1iDM20Jf-&Z7qfo-_b#+m5zJhOfSEC{@tseYtCkzz^hPOu3IWRhs#y$yvR~MO&oC&9&#n@xJ0K1m(Ne0OvV(ChoA%JRSPcQA>mwW$E`dpmJYs6j!4g(PZY~K86h^~LT{8+22s15y-YhkUgtX;chj4h%gDb*>veC6tx3Ngsw#EkfCbzcPCMY?xQnFAfnWp7U}Mrf-t+TyCdn?Lt}n5tHnayeME+D;NNXGQgU$luBY4d|Pr1c>GN-*u%#}>(Tt-a)a6Y7UsA=EMQ?$^&y5_iX2k>sIYJ?J?@0EPeg|xiZOmQ5JV_mu42eajc*yNuLH|pmp)!@!=1Y5qTk3JoH)kqrbWy9lY&+TnG@@DM5Z4Qi?f8t>!hrafm*0fEGPjOAmP}Id50|Zv&N@O<9Ak{#Y%9}TK6!NY4DJ-|3-K9Wn7l5cQwd;OYfhE%n_7helmmzz<4XGi26XI$D7Wl-G<*c#2v^i<Fx!Dczc43EYyIZx3t3D7&V{w>3JYJX&)fOq1f@g>mEq8{4m6hey3n!_a+%1y6)RtRKvCs(t8Owpf?D7}Iz-Kq3(Pr!Sbv-~v1FTa*I{0m;;&p%z2PIuyf!kF+RS4)yOtu(cU^}~Yk*a~^yIiSQ*HUyn-PW2>mILg~rOa+6xh+|l=^_P_m&=0}NjJ%HtU&el#2jAMP3dtyk}dO4p1r}UY;Y%{N5aZ#zOTc%S%;fd+cTdrL|cPj^10yn#kwQ5yH3f@Sv+%U*CxYNZadpGOH2o|h(o^+3yN0ccNJ_UgsquvmDi3@sq<tfoWSpzlY8{M!(vE5kNSMpJczr+m{=u{vX&OroLose?1Nie$nw)Ul@8nEuD=^C^3a{zUDwqcaM2D%ra4f8rOoGFCSZ2gw3@ZohN`R4LP<w()u=UcOldf*j+*vlMCKf4dZsEkOULj!z->yu3+vKB(qGAht%Yt=6psG+Hi*)(KCxOZ-N`^X9Z%*eAiAu+6Cd51ysto&HjmWou#<O#O{GCzGsFB5ZF3nRJ3fLjHe81*wA7xpnv~ffu!}cgy-cjs<!!HP?&@QY7>rX`5>Dl<)Na<V;|R_7D)nJpO9ml#EHf6ds0}loWNnP>Y6s=V_uN)HcPr2QX1?lI!F7(-DKy*-9<^v&>+YZ1Lv-zC7qxtIGUNc}2CjzGbfi61CtN{ef0VqGm}niw?cRRoPGI%HHtn$3@61P|x;3nyi!m)ZV~R<LprG&-lYeu7dFr^0O@-H_l_{ZkG+FabKi|Qbd)8M{iLXr!)oCYhMCa1MI2f||OugCQxsETe1sj+iK$xj^&W&V6W;i`jz(+D(*=I;QSQ}tZPrVR2CYyWJ-`{~HNmbo$)8BV@usYtR)tfP>a>^T~$~7;uoGxnA>I{(&={8Cz3w)^kx*SyYGi9HB17ibwmKtE`CeKdDi`&fGNBMq1qvB4T5cl8+sMwxwEb7x%U{xli%E6lrEuqA+90atp!&0vxTc`DArcCv`0>7z2N6Q5z+`X??7jbC0eJpd{nl7+)J-}OblCwIfXAVWxsC62<OaKp(P!{UpOXWg4N#e?dt}I)DJ_DdLz9AeF+|_z_KE}2BW&WT9y$2rT<!CrLHw96!vN+XdvX*xr0v8K{w(w#Uk~P0dJAD00L8&J6R|BF)QT<rTjo=Gv<Z`>mNET+meb(#OTckCAMS6cz_h{Hk?!&I8sQ4(nYn>2yIJaV2%`fA$7mp{~`E8+Id5vi&^xm^G$(7wMifTF2tOiYKP!?PL3P%vk2p9~Xw7)y5`22G6Eq8FiSL5bs*st74ecKjfj0(54!})ry4*l%F65P-fuh3$-5on=zB@1A>E3UiEWgmp_J``#M*r{b@h-Mkr>XGsr$nBD=U&}gYVjk9cqF+=|WEIvaGMU1A<VuJaSVvx?%HA<I&`9lhwK4{pqdRb1NOU<EE|V@jRI;WBZn5lm29zXsb!IJelXtgDYahlC{4jy^T08HV4pScHyh)}4pYcw3Z5CGg<8XZ)*-O>V?RL%x4-T`vO4n{OPA4&h&Ya1pKftm|iMqXfq8%)rYO}X{1JfH|6bFZAalAfRxhn%+wDyk3E37>*Y+&TBO1bE2`Vtzy7BFUUVe>}ZvQB_YnGw#T9c_j@YiGQ;E|SLJ7?&!Bp_K7WOB-W1vT)gz5)@h#qs+7BnfU44=^Hw9JE)33H1d+R4&x@YcwIe)0rIq25e+a=h^f@_-=zMC4e$1SxrBDbySIA+!}6Mdlg7OijA=ZR<c^vGZ#%jLcbgsA5&br)g^{2X!`76$_rx`N-MS@!wyuu$?o2(ebU#SaCWiKxVmzK^S{?VI9jlTsE@*N*u#T$+!ty&Y%v#Sqv^rG<EfB@=ro9H;BcYXLEi88@w@12p6}w{!q_46(t*c3?A(U=xFW+VOqXP*6QMLPdcQKENG@Ua1mhRm7rEcbpXosxUVl6ru#bK7aZ6>u|gDN?ACozb#=?;5uWP@Kj9VTY?7XWlSo$0poW^oI7v;w<Y7UlJJS<&{ShL2hU;wft{&Y)_6aFj+#c$rU(9J_glCIh+vw(1pKvUja|L%ui6or;u-bXqo0HQ*H(>%kRZ465WJ%61v4qs6@$Hs;-ck{j@qpWpX`x}R&TR;TB}f#78`&x{M{J!7m#s9$p5{iSfgxA>A}WvG;$j9cYKbKF{?i^F`uiG(GvtkK(Bu9nMY9^3X27(a#$iHV5O6Y8GBv_ld0IF|E4<H6m7bC>r#_bKLNMTQboAFn6uwC>J!F_)`0tx43c0`ep?w=I1=DIa?HNe3HOro%zUYk4oM7q#(iuCoX$WJd`~<e%876@d+Lx{QN-=~1QX(D>2Uw%hBww4kUsxGsw*V0F90d$RM<fzzQ2<(oExOnGl7)%C%D^Sb0925jEigqHj|weF+GtsUJa7JsRt8I-)`C{hOW<UX2<)7kL8y>DI}YULfchW^4f48D2pk4xBL1wFI}=ynK5T|(x2T<Lr&NuF2+$4YZ#wrcp0-)3-f$5z80e*hoNI<YJu%$@7?6NDN!rvWt!su(#Cs_vWs)ed1N6Y<&(0J-*bi}iW0P;TP$^@LXHYphqpRx!ys&OID=jR8_TT6huM70#Wpba75@A$>!TY(_4rD-N0*UVCr=qVDz9ndD|N9OwcY?&syod4WXMW3&fAw&N3AE%%1YBol`Dh7pGhx1QIl_3dtN?O<|O5}#K-TYSHv7cf0cW>wX@;C(`Jv*)BB<zI&_X@-RPQicZGA07n#_$VjY9d0SAiXUu1q3O%J0!E!|oH8P}(Tb5*o^`Oene%Ys73&ofY5O#tib4>IYG+^Wh1mqS)oY=VRHq{XMH7I@Vm<?fbgj)6%NviV0M(v898c}`!%V%c-hbV+f+-qxqIf-UW%dxndUA3U1U)36{A4A&uF*^24UtAwesyavEg)tv6H<wG`vMa8i?zltEWF1045+R3=bbZGE5yH1Sw4S?4wHEcS>=N}ZtIdToz1EzmDzU8mAps&mard+wUy0?M3NC*N{4gVUSr9PuUo769D7nF5NGAyYH64Sd^#lS<>WdqcE!V*BY}ZtXBPz0dQ3|2c{KAjRVdD6XSqrDIxfTZ8=5AC?5y0$j{AMDb_C48y%nQwkZsh9Q<w}abVo!oCYp4feB7($i8d~C^l0h%ZF|_+w)>|`LqbEQ2&E;wBm%qD#NCBww*j<Tv4I!bz4^$LO`c}`^rCgVk#*_bm_#%d`bDI&xmOxLlli4KDvXlGieD56;wCky_Nb_3kS#Y@YM{`3_0w~ZvwLO?7M1jVcS7(govy9w^qA@NpT60@%K=A>=M|>PKKiA8ei3rmWk1d=arNG8;{KYhpqr_6b3&-T9<-}cwq%cOJIl{$Y<4R|k%P^uQ3zF2VFmU}A>+Zfu6Ly3y<|mi=<SH_d$HN_$8KX{l18^W!7|GpN}g{fYRivdcATvgyuk#cpeEo?Oy^#J0f%@S?6fOF9el)F$-j5{r!CN}O&1+KTbU3gI6qnBx74#+HI2z4c@R4ig~%>DqV|V0eQuh}yl)-ZI2X2!H?3W^EZe0Gh-nZaIxKMQ)-)CUHvxNuvg7j+CS~uTrtiW*a^vCFqHtXFns6x`Z$N1NLK#IO<SwDxMI3uon|Ew(2j^BBNjKB|ia3L@>p92Jy4aZ9yt7jV^=}0kb%-s;-YV&9k=IKFiCkxg@MeyCBOTf9&Wd}boBMh!r2MwIdgzSXHZ_i@^|S5U2)$;7yqb*A=bH|LnF81?hQjpa7)MVv77wAG-IB}P`354e(!6DJQH>Z>%H$0yF8~Xa+V>+Ln{nfu9h=C;JbHuWAe#5?%UUHazfrm{q&Gb`kV=BeG&TK{j;E=%Q(Fhhy1>3?^&9yYR$5;CZPY$haYWd+16A3D`zvv%aL?|c&&NQ&Lv&Z=v&=T7ehgwSsn-k`byft#9I$B6jOf69HQvpI_ApJ6Iy0DIbOA?dR+5Y=vbX1pI(aG<0r)g#xeM_SYQtI;xGluOmY7@^<<V}Odyd`YFn}^AOT)Er*ULBO8sF|7T*R>PasTE&Bn$(<##3un>ck4iGl!KMZT8)A>x~T0<en2&#r)lUhUfMnnys~>2$8G8t7i7}bB%cPd_Q_}57!uT1Mw!@mRLv^2XGa<+WiADQ#;EA)y-rN6gC8w$wNUvecyb{mz`y6R-;L6m*#6Y=3<32dBLe^qa$BucK@2q%nW`sdFF`rE{Ye3u2(MGvhvD+`3J;Eb^GcJp6_I7USoqD^Z6`=?RU4q*vTT88sja0_=v&Xc$aHeFHN{{Lv2aYE5~9AL4hLa(D5K;_MFxO?ExERPfx<yA8w<fh3CWT$qYvQ^6V0;<8?+%wkmfWU_tBIwxLo)o70)Z0(y=+*w@?hwDwQH!4~ZSW)#@&W|~pP8-*>@vWGjrh}%<fC(efO<Q4c}MKuTBs|Dr;#+ayYAd=YK@9Hp7y0_(4mSn323$skUZaCYD#O7Ds{cE|KT`&>K@2<XAibo@eNIL})#{Cu7=fKLOH*=p;_KqA@0=R@~wUd%{25Ur}V4HO#!#D-s*yCiQNJ~u1is@^oh;K+5u3JDgm9}IvO|xu{x(3H8EZnDM?-Wm0`_RQmtkZHAhc0aP>TPWqssnfaLUeFi!nVqF&0N&nHg3DSrh^`5=%wlS%V!Bc9`v}P!6&3KtpSUi=;#U5UOxAWDK&1i5m!dfhE&>@wgk!E#~Du-7mK$=z^<>z!m(mDsyWKp)KfE!hWJx!s112udbarKsZ+`3q$`zkZlkEN6=j~v9z^I}I892Q6+n1eAyQG<$@j96dqOn@GtP~EzDRI+;zXVD7=0-KdEqoL$12q%sn_0b?uw|>L5bNA(TQlET@;Bo*jMhAk_oX~cr#StDRfr3#+G_5<|TVMZr+`^9|7-%jd)Ndt00vF-P^o26fM+GS7_Wn?v6lj4%IJrDG73bd?)k!;rM>b%(FI!_m&m#thcW$;E40;PLa%s=v4WtJ7v<UB9o2EIk)#Ud{>ObRj;CS=L~WH#~^dFyh`(wo9<6hk(a|3cHYmM<HUA13;Z1G^Nxs9j|P-e8(Oi@TDxjxg}TQE))~oEq~=$9@F;8UyC~(eIi$pkrs;O%B!`=?UYrx}P^1rM-Zh;Zh!IK3q9g%_rngAU=2~%UoUprW)3O8A!B#Lp8armCSmNbzc|j{>y~4KOA*bvyd|d~;pf@G(jnq!5j8<DeGl$boHfvi!n)L$m=pJx=Kd!`h;22{BK8dh)8uuMD_tZ4&Ebn9cDUM>~m~BiX*vUa%zmc1petv#)7ZGvOL3e5-H`uZ7wW7p#7E0s3JzQ9(TxeM}zXtcFa-*~IIp39}>^$3JA=+udy=b|G{O#o&0d-kyE`3w?MW-<Fm=lBXkj7_<0*61X8-<sz$MZzj38YrL0ZLHz6hVy53eAO)R@vz1*K2iYfv5AUPqB!GF~^!Lzd^_dtqiDk#2<#!{+)gl;y%cgX*7S}0q+Jhe1PsiW*0Pv;~rAp)KZRRm1j;>qr+%lERyG5oVwG^X*NptW2*6PL2{PT5--cL&@?kW2v<LE569Y>dsLhX#EiYPLtN)hY2`gr`y>x4$NughF37w(9IUtY>9*-F_Spl!o0bE-2A2D7M;vH|l-wim$?itPst8VJ4Us%uqz2B8-yKCp`uKc`nCUQZt0tIKZvJ`<b3tllVh(OjumX`i4j77<b@neB!sF4-R%%7yEVYi2TN$2keNkA}ddpWY$wZxb)yYOhBifRB#17ecXT49Mzy|QzfL3vbQ#1GI89XkBn}|N!<cu3f1E+h&d-;YyMn%2V5kX^S)D%x92(Y0S63j#<(6)Wb)gN4iSZA{Bp(n{b;L?HtE^TOGQV&aO^yGQsY**Ux=roJM;Th8oasPfqD34NT8_D&#`i|aZDx{d{9jF4%W>6p*W+zeQR=2nA6};g&s{=sjXk!Q?pX{Q@%zx-nAzD^AetIYeTEZ3<(t}A*Eq%e<CDWQlD0I5r@uUeyR6nO5Rnm%w^T8pE^u5}^B$g-#*+lbCIIZNH1P)y$L6tjRbjr(?+LYW<adc_J9=xW#-9|#6uvXOPh*G#2ZV#h`^m&QxRk}LHosACMQ{MtIbLiUj&*e9}bYckh4b3}<>2}d=PrKTjNA_<o%{Jtt)}}GORzNZSGHM>O?JDpn;;wVkkO}UOCaY#z%|wTi+jOtlhQX92szWX%VtJ-Ht7-TvqL?&GxBVfvON~lRlLIK0O1)?f<y+D0=;;lf2e|EWa3%8Kt%Wfxa5vKqt{<u1C9j70Ws2y}f>o@&Zo<pT&iN9Z#npBQrRc0#*&S|UI4ksLb@>tsd2@Z=XGgxzP`g?{(|wkB7P&)D2)X`dNODJX|E#F~w7%9?a}`I*TV=j^rA=>X^nikko@@I!zeg458uGbwn>tLBGdMc-E1P1IGa72e09R2N+W5FR!K3aKo;9*@aXO3(Q)ja8?cqzuur7l_1AHBXiNExx*nGnu0gVp5IM#Keea==M^`yDIWyP$PlK5;mxeoAh=XS3){YiD<GL+vcc0}VjnJ<0f$(!P!Hl>!Vx<A*7Iifz8Tl2vp0)@;Dz+KD<`2!TjGk%v*irOVuo!;zuPpzhhgkP-p%niHLq|I=8IldVU*-z(c!<_`etiza^9gGXqbUHrk9)G@1*$!88G`2x3Q6sY`I#pA4weGnjgqc-uf*Y|W=U`gWtI{Q3S}bcYVPk;q(JP_^G#=cxl);TXQW@UCF7mw4ugY=84kVzfl;>z`O}fh#vNY{%ju?$cFI=nUN;yBCmQO;*^t^HcU1M^xs8+L?C9;)l^9W+dgyEad@(Q5n-Mzox%9&EXUT%yPgMSv%MWzN{8^nfPI%mGNwXV}-1OY3FKbG`o6MD2)S%y2RZ7e_H^~41S!$?<M57gtkD(%4x^vdkLkW}w;d*&lzuE1b0i0yo{!iJ+~Z%EDZLDYgahaGlwl!3DYiTV-Ln#T&2hS+rtWK1^(7V{2#?FPFX1URm{80Ddhggi=x!Ki@WPdNjeHhUp(a2qa4b9|vX+j^ASA0=e5gWd+2lX)OZivM5>$u!u_pTld!H);8J9+16}Aq>N=TY`iW=yzw{7ImC8aND$yR|Oo9cL`o+@G9xjxtR>v+@@Cq%woVym+PAIZvQef+h;0cj#w1+X#RNSqDr{2pBb-v^;j~aYv)=;o&_(p_&Al$pmlg@wOVjnyy41zqYfHZ(yG;V@~W}TU{tf0`Uy7Sy47ZzTC8dV$vh;;wudzfz3Vi!AgB3!^DnyuK=uU8V&{PNXGejYkamr6%)@?AmF5+IfujZq`95Vw)dIZMdgXPJ$mn{E--SjJX@j(G5kzk_XV%t!&6aZyI+kL<y3mUlLZ97qdJ5N3Kt7q)nw4v&IBVA39HVvCja=yzJa6p^!#2fK*(0eozC6eR->Jydyms{n>#*3sonq@A^OaG99Ov@|66{fzlZiZzhwZfmZ%Q*O-aRs<@vH|l*|KH8aPw&r`34ziX(yLGZJl`&xfjFzL2DS&Q=itJYBM#veMP%eMbzMqfKY4#Y7^YaCrdR&%k4EDhZ{6p(F>=!vywYk6vKI^W13@OI<~guQ`NwR9lUROs8~o3*Fe<n0GF2w6=#B6veNL}t)kBIX$xqshYMH%WmYf#4kLQ35#lq*Jy=mVB;7H*#Yq6XoXdqyJb14N)Zc`3c<VNbSA+`63yBKvV*tsC)w5OHTnlBR+)S{`HNU*>@bMA82=!~et(}#zYm`<CPafI(!c<RRjp1v??-Bh_1$)z3Kkz!cFsR%^0PK}#%q%SqOcG88t!B2}pIk6+OG68+!A7l(Hn>&u=Yv=79trgpP3Ph@%a#IWbV#PlhZv2f)rst7rjtaW_cWzUp2>~#QR|S5aNe8QMXMK}`Lypb`?x`gFI_4?apefIrSr%%#8dO!t}S>7M%dTOz+kv#p4vn$?-wT*xtHkk7!3MBy4+WCR_@N{TU$4v<z6c}2neT|l?P4cj@M_wHA~Z|dq5dl>`XFMv2edbd}&bMZJBc|U*;>v*KJqjD=!6hsog+Q8tVZMXX^6g$&n`(%B~t2l1bZ6`MpI_ujkCDt340PEky)};|T4&;&*+>URw0F8?PW^T2%mS&up47dj=iC$kMXz>EMnBC(ZJ7*k;>f=`F;8-kPR$!Fj`?3$PRqt>?KgMxjD(qQ2JXG;iRg%WXKd`ED<~7P4L*9SplzG9vqMq0gMsMTauh5g`KlhCT?yXj7q%s#Uz=CnDplx2JJ_O{v$jE@bOR0`b(#OP@pg1u{o1ojYLmgB>i&Rl`mOfF*2IxdC_En5eCbO<tnc?c=rq#wD{-cn0<O46U4I8)sh2E8`<fvM+&a&M64nTva!0L%z+3!US*Qt#Hb3pEBKj(s%AzquPg<b!z9aP=|;Ct+k2_Ogv1>vyZF&gP7+L_p~&^6<v{xRzX_tn;D_GzW{ufDVU>943ubLGJz=gh9Lk#7UgA`gBnlcg)Z8b*wo<&^NZ~vawF#+u*<$WcdDvy7evqs(2C-RB={H!`Sh*~D>hC}b!rm9nnefx!z|I$4QI>-#b9Y~$Nc^hOi$#lO&UAJxF0)yoTZ+%TFaJ8;Lz+7#DZ`5bG|QO4(cX7Ut;%u`&EDJN19BV)=OpvlhuK}mr2(ahH<tfZt18x5wP-jea+0`we~0%op*p{Bs+hk>cYhZ^LTxUQ{H>tVGD2Fk_O{mnbk=XW^>=|6#W%`gq$UzoWONjsExt+=JN?$JoT14tZQO|-isDAC3GFvI~o%q(U{1zayL(0ye#^TwMN5yepHh4Oi`0hwZ}G(w1m+muD7Y9QA>O4mzk?m`l!s@{hD6r0PFH198b8yc{iG-^Mf%lD?C`kXQ$m8<Tn(}-(+0P8V_zTH}fW`&ukmIz0Fb&cco{z#T$q0M%8g@A~Hus*@qec9j==-hs+izH0)%YeFwQ+m79M3mW56oYE38Sm$0z1PZM477^}r(lA0{ZU1aDvbEVzpWdY+pdoZ7RP9YyVr@YB;h6AdE%w@%KBCIDbpCUW+kErBBq`5x7OVH#<T{dYP1z0p}GshBH^VAv3^y#2_e7vcuJWDG%#HvT!#-Hg)rrYfSMR-Lbt;J|mA}><LEBSNIo^a+wX^hFh&fMthb$~6!I9wOEX|W7kQA7a!r0^U(nR!ac$g&p?5r-{5T;o_@3&+-JuLdt=H0NGdWLP*~`cL%6)h4dp>#S4pJ{#4;3P#JPAr`iYO&(~yIu$Qp7Y)df-O_oTFRXym<$zyGr_N5>G_QjTJ<n2%(u~rGIU~9>pGcE{s}^uWF}iR+(6H=?j+TA2=U+=H`pjkE(ZkipB0Z7xX{m5)oszlUhdoYq{8q@u&oRoi_U+vXqrn?=ZwNuVx>WP|u2XESY6~ejDBf}b>A{?wpKVjG`W-LI^wCZuH&|EV^fY3o2TdDU`$gMTZyYSuhGoj24!L#v4zU}4A#`oHo9QS#<j-<xXW84e5XFq=$MVf??d!XwJ;0H6F<XD0D0SZIFJ{f*j@bzf_jwHl>sn!KUS7Q8cycV5@N&KF!{>T;OBzdevM8Thj6S<QFv17CN6T=G7&e>!ctnBkd~iCDCQV``9Dte8A_L4#BSW=YY>ib9r}adTCCuhQb>yp&Ej`ew7i2WHm0yl-)n{`it`uyD%vaA-kXBZ12h5%|4>#N;GM@|+fAK0WU#oQ?ANl8aUVo&U<8xpf5$PCXHF{M#*Pdw{nBlfv$rPW?rc}@tVjzfd*^4&SsDHnR=FI3tkkhrDs^M6@J~zLE^)4jO!+T*->MYjV#9E$*7oa47V}DT}Ca<M7kP7WsJkEsPFpREAk+-iIAbxb0D?Aaoo2ub-&H&~PbidLW4R#}pI7fGMXe4d~`v)*Dg!{^IFKF{no8Q1yzlG!uGJB}QI9n-_P0ByZP1`NAElSVf037A6@HKDH_0n=8M?|0C9>lX^tAarSIStx-158v)GtjDd_R5N+{iW@kyR-RL%a4k^oN8Tn@47#mSkW`Om+tCN(@v_K@tRBY&}JKX=~4y#`-FFT6A0}y^)<D|h6|)EwF|_IlSU#-tI0f0=Dj9r3E3BTn{R96ZLx$mbjiQ&T2xv!yXTu)9&`uGLfsu)TfS9mP5F_rhj2pAwz_(;xhza1@#roU@mxO?!}F+^y{h?9*p^i1p=ya$-QCjg&>55KM6Ed9W5>ZzG%catqtY8k%3Rd0DJwj{7dEJD)yy(iKLclOT`S*``R<yV+}mb*y$4}uTd~`uE;Z*ZS?iUw<2o?YOEHHlOp&NUT5~fM*IY)}gfw>M4!Sob3+4TO%H&u1es7~r%+eLdMz2V;n-E`iX6AIq8dd)Z6IB*=5^YDT<6UJ~K{fR9?(?HwFD;(OoZ%y+%v#8};$oV8d0${ipKP~kMLcL1m>5soWYro;$+H5ixpr~m@yiMqR0|hyOZvT)xVRZY;^atU)9YP&+3t`=8*~F@_oq|6^4bhTIwwWt5qSp}$F3<KTlam59So{8-BW?msg;{s_kcN`W+1*hov4==^{V4#!CAxZHiq6MtToPo4QFwmXNNY<6wdRiKgbu<)u>}!<!6!6clRy2FTuOje#qIiTr+eRZ1s_4@iK>HU8}L|IpthwLW8O0X9aq<PJ%-DSQ5+LTwPwpP7rg}MMXwww$<P_;sJbYh`<#*BlW`#z|YNhg267@gVVudy7!+0#(PypMK%|mE4s^H9ffm(ocFB4wM}FnM{>QfK9Z<J+=mCdQmUx93@cB}I4g64tO1!kr_N-ee3?p`g6drI#(h`Ro=1gxMW(B4v<GL6C6rraN0V#SgDR(4nmr(v3Typwx8}!&>9$Cg=w*lPUK>!527++6!L#}x{bzmSq}BWzin|Ds717>}X-ZxZ4G!BZbidb5bu<b~>70M_&uPxA68p~nLfz(z)NGB~eFoe&8GgA*3BXM&aRPbyS%}t8ZxmnH^+DH1bJ>v<EL9TdCu-8hF3@0Iz&2Jy4(B7Qa80uPW6wHm@+HFSjI*6l)p+{!pjKX(m)l$+*W5~Y1kC^)_J%w#Xux>ZZk}`;J)`X!TRxPF=TipXK-KVYwXcdhQ&wo6Qkny&+#3xZrTy!;U3K_aX-9)9GDuRo{+Q`(%f!7&dz0;*n)$(4E2eQ<<i<rKnan^2YxIkcSVoGCWw$@>M|K(SMksQ7qcU{zhEz4MS*L;Q&1`d(QKmj2C*anUw%s=nr|#5pb(eVoJ;bk$F7j%pW&|<QX0c~(@CGep@6>p6XRF@z_eH6Qru$At<W4#{rv!9~jrV!sob__1te!+I=gpK+2?X!Eb3LB6>Y<=mu)CP_IXoj1Cu!H3m56(?P-inoX<aWg>T+N+cp`<Mw?cC{b0uD8jX`ec%LKQYm4e3AA4}8O66tl>C@d;V>P%#5;o&f?!Pc)BW^Mxn#-f^<*h{8c?@*GlF%nA2obUj)Uu~8VXgLJ0l<~+c9oKH8>snSmS8(PMP>==~)Nr{UPRq|usuo|=C{s0^{5BDBARV{iM0mX~XKlW&l+$D}u!lyVp^!bF?<ak6?tw(UcTUYz5H=c_@Na0<a8zv7v@#n~LKNjqnAICNIpO=Az{Z3@FEe?0Qr+ys*~UVs+~&ggqcw&i?2755lkH$8NKWX7-ML&U%Bu~VXIG`pt$1AN=yHo+gJ)jCEpx{|+p&>q_q&opBWRVgq4W*PmeAZ{jspvoGI_v;TdxrJcWW-&9KJjDrCQDqU<xXg5-8RYOmCU@(Ia@ROAgnyA&S}D_bFLI__opn_In9D=Z~ldGpyx^rQD4!!~5)k>h~&4lMaxP%dD&+b59D%8Y%EtMg+U9Qn`z0Na9s4ncagv_{3xnO=~a&6`vZ?0FcbeD5#?{WIB!PDsu<<8=FoKrY+}MYQM0$5V}8INl8NrGe#X86OCsDCH+#lbZA!dI$<z1es-qwvujUDUAf$t5nfnb<;uZo7dUXf1KZgRn)G_{+<YyYK%GDu$NH;D24(z2Udp_hl&crH@amHr3$+*;J{lKdw45f47*1Awd)qSMN9PF+PHvNKua-H_z=GK$%(uN}@Foxscl&q}wL#0RZk_A8*XOl<vR;G2m!I%gMaQ+cl2uN+vo+Rb(W^dAdy^Y>WlXcT**AqOQEg)jU-!o)SdletNg}dxX6@aUxQ?)N>_{vK7-}tgbf>dv7K=J^P`qA)o1XSHXEG@t%VD$77%F(|2<L%BXEi5Z^_S5wZDrNuq4mI6@7J$S5rJZ(o3=9Zfz00?8`jkvBje}S^$Y^JB-cQTFapl!4Hs$Nxl`S5ojV;v!4x5=js%n9^l@vsXb#*6<g;Ka4fnV_$S~b8&gcn4LC+jD9S8&Kjg5=d{Gz8XySkJ!k=1m;L!*aYbD0;M)}LdeeP@tm9d1;oqqzbq>>GE54%TQe<wAgUEJ+nR7fX=JZ`(%D74c12zoN}$IhiKKS?u8Uau}g3RmXLwn_<(dx_C#F2{kpn727pws}+y&0QL^0i}*%w@u@$5EJl!DJYx^(YVUNRWSm|aHqMzs$3HLeCwy+&1-9q6bNCDDWY(IMnNC2DUUeGJP(5KKC*Ck`p0yyUp>40W*1mnd*Fl{hnpG~br(Kr4h3UJu`^C|P>3auaRM&_3EHMb(m8K|+=xA^;jk>e6?2aZ!yN=i^ObV#d!&dw2y}n)7SFvNCCSnGmy9@GO>fIWN{UXuve1GI)zwb=eg+hh4@{1x0EsLjDZ&C>xY~8<PH^{aem0vfOMA_<<s-W>+Jlv{1XzJx|*QD=eBxal&Y!;<1HLK`sJipk|Ca3q$!HBWZI<vT`C$f}3L|SLn72wVRv~r}E?Zo($rR-e4yaiHnvAfz#{lSe)tu-f`@hO8zZW`9=<sj2vcW=&B^18)V13r>(l31!OJJy-5X6_G`8kzzq(i~dg#oSzxrl-tIDiyPboYXVZR6lD6Z{m)&wdtDlt+-UVZP%OP)Cy*u<+(qw>KQtRi9-;BE;NoMFm_xHdH&Ex`t>doh6r8HvUx!R^7Be^w`WP#+@0FW@xm4-Gpc2*qw-{P-yOLgjHjoEnQ$$AzjCG6Zaph6WjI$Fra9IsSMiZCr5BWRn22ggrF>5A#QA}_>esp}5XW?+`ce&j&X21-o-lBiNVff2cpnsV0WwKf+A<N^r^e$<&STF6Zrl9Gyjz4CP7c#Yz0!AT=;q7MTxvqK;t^_m*@#{NA<IvHvaJB;>1L}0ueOT^hx{0b&Q!KbUk=Y3unltHVwY{>>lW$&nc|e(KXOZ+cR9kg+fKs<`D3-R@3t&wh;;hewo&SXfH^)dSjL&LI{=(7<9_j+(CjI8Y83+C!41PzT)yNCE31Ir%`&gd;?m-d3cM{<6TFi9zo<C3rgOnHjDMC!88K$E&uE`Y6jF%}O0-c#L`C@Q>;11;C+o0-cF)}Rb^WfV<{op^OK(1hx3B=XN<}_B-yU<Xy+BH=o%QMAT76~aayYwPs5iwH*>aClHcc~r{ROGn>Jg0C+x3Os8VzI#b-%$8omJ}CrDw2W8?B^$jQ$ftC2msqZw{a4t7oP1_GPtv<5TY0|7<X8HoE~U0v<sxnM@nVxyrZas{88>;SYT8y1cy3$1eW!sqM{!X@2KB{Am&H*t;4p1_4o4go+k;G_Md1>APt)t3TOv{9rp0g*4uY6x>qETiHHM=FQ5>>HWl`k{-BQATr3O!2K8Z-3j+(kRL0TXvFhWMQ9pT-mUD6u&ej4wVnO?!KUjN$fx^^f45o}+%V@)vDt_D?$D)<$R>Z$Be#SYCR$_t&()eC;)<4-15g<zla2YvTV};>w#vuty9~V)`ROxxwLXrH{ZHMAK*;Nf_IV%ud`xcc3nPxU9mskQPXeA{l7fRl`8lU!%>X)@MiYDfZtqHHQ%=G;KB3mpWR9EiL;Kxi)IjF`r|BwhzEYoyuOCwk7L#zwFPiS4-)oNE^@j%ZeB=C_@jd*v1#0o-F`nv0*Xe3hdm)<LjeZmg8CM!dPOpRd6E8M5h~6p{H|+GlcKgqD8yDI?o6mY0U!mzkd3{I^op;2K+KdYI^%zQSbyr2?)n4+q<C0A1Nuo@+W$A~(FW#h{-EB-)dAQs!CaSzVaI>l7^?RDK8yxwSstsgq@(f3Hf|LIG<$PEoV2S`?6&!v~<fJQvkW%U06{0hZ#|iN4S)q^{=FwF)*puhs-AMdd+sU-a<@Ne_kF0=oZXjNU65Hatu1yZuXM3Ko<9Aw98V_Ci_4|ywseg8NS?TDVbAI6~(K6;$*!;c*rxpDeHWR*B9p@iq=eMe1L#uqoG(7ccOt}W$>^)JK@#gpOZzth<mec~jEMxH?Ca`TeqX#u_b+Pfc?#CewZhFIEYw#eH{&O`}8@G8QW?MS5ci|5;^f%kWW7S%Hv{$#jNiX{aTnqASj>Z;w>s)S86F4(4%Jm*<XFZ_j4av3Y^fy$mISs5oKYkcfU2i(~4mQ>6!B#tzPrrZvZ57Vt<XtMS2!`-6yVX2s-SmC3^BJraf>|%N!B>gB!x>;3>}j(}(%ZFTB$f}(fR%6AuNV;2^R%jNzmNzDG}Umbc@32Do29jYA=iXDgScrFGl_=FdBWm;$7A39+Md*jKRqX20~EZ-*!f-ED<_*e(Yv^q)XxQGl#4kw#fE{F9eyyhCNBF$ceVDbt(O!Y#t;0O)gp<uZ!aCF9)e|PJ#P^P`ZfJot{r@;pI;+VyKVh9tAjy@`z}v18iwJiZ;bH9@t&?}4x^{N!K;<Wz2kw2iw2dF25CxYZF_?FNyp1ut82F$Jz>M_wogP$>^cGuloRgwZPd&$(HYv;+p6fvfH~_=OJhFDM~#Q#74*Gnm?86yYJ<;)r72}e7SD}4t~E9d(iHOQ4S3esUB3X}#l@-LWuQeK?v2{GZzc=x#!TFn@;0Agn|n!zn0+zjrg>v_%~^+Se7x-CM=(zem|<S4NrHAx7jwHhn~Yx0SsH88JA!T!>PgR9Kp@6&<;w$^*<deF@tnE_9(oJ41F%kG8dr-lSjHnQmw&2`i8E()t!5nhgU{1=A-{fp_1Uguj5P=Rcl54^$1Xz|U(A16E45A{+<<-Np=54n__$MLJ7iC*ew8IKnE7Y5+X93V^ZH(^D?67_^Ug9fuZBkR)(|mTUcX))tVh^$n8)kJ`ouT)eOMB%J397S>}t4(R^eTACI^)5j?w1ki6gq4NV6p{xt7N=;otICJDi}52PU<C)|hd1)dpwGpMa`8;;`@R>&mZv`pl->RE5a)q+Yq5$r!pEPb1ocPL2uFy{y0GzC0(*A&8mG-=9a*3u<q>m|-(i%82eu96HOJIo|c2+qvF~>sKz3HosRhE0>Gs6j<MG@%Da^##d;f%`h$R#oMZ1Fr#g+O3Rx>WH9aL-#kP+hi{4R4#Wq)hnrIvXkZ(1(~k9>bK3Ui$8OU`QO=8+dBU;C1?^<;HUQDhVce{(gyj%$5H+32#^lq)$1z?x8IOoW<E5I^TTiezA6UEA9Ct^={o)dFwp~;HRte19!(Y@ib4p44s-sS8kwD)+Fn(OLCmpPM6UaD1+wys;Oj&IoRg3TGVlMVlR1uY7dOYES;DT<tPmmDjYeTdi!|V1)^yk@T4!uE?j&BiUo*<kvyX#>`?;2#%CO~Ow;Z&ZIQ~k6&sCy1NN{@)SfuJybRh;lI|AD9FQSoK++~I*9@(DEkfGsMcVvKhW!;k8{u7#Qialg*M&`FiI)u9Mq`1Ac5gMQrych|Dg5!{m*3=Bvwmt*cqRR^bYtv-V-XFnGTqOX7^K3$ddL1Go4T?|Ff1+&ZW!~gV<C6<4nJUnEhJYRog7$;J`&o$-~ot1&QJ8p|L+TL1sQ#;1*Z~k5d<+>PLPFHh|3HgO_FT{#k4ZzOq<)>axYa_-6rwWwlKR&$k+_#+dNovxYhx5*V2k5wVm&&}6aci4=nIp_<*XdGkGG76#f$z&~1ynveJJ`vG#lPLYJ?a$AWz4BvGF{&KVS}EZ!!-GhX5OJG*7fJ@Yab0AZC;*xiX|W4XZ6j3gRwg6p5c4-G=Y3kdzp5pu{povyAL4m*5peblegKfI)R>7h&XBVc8ifX<Y23}xMUG6$fX`soRmlm)7<6&jZOJpcMBSgtqq*q`aT!u!Bfbp^ySV<v%x5t1AQB-0&j15pp>@k<#4jC5}(FP?V#<XL2M!MS|e}QaSXV3ukL^aIPr&j`PF?kuZ`Izp1dl@H?sG{ZJ-yU+mBmcGN*?c%@6CA<7yY!kJhrt<jp>MWha}am1<fAp4s_(cuzQRH2h<cNjYdE=pC-%`?PbR+0Z?2A^9Z1%`v{|U6V^mV6V!@Kj{w7m`t6o^Bl+R+^O1~xA(hjQ%5Iw*LKI&lSo=}b9)WWB2)*IfeYVQSir=ym@THnr#?qe@jMV-mT<DR1XFQ;;3s~V@19ydQ^@zS=a_1b`>m=#mu&n(eTlB${J7Cw9&vfE$4{{$aeH++HH#qbrz9jVUFrDOm)!e<X<79~F~p`m<t#lM&Hvokk{uww-ycorYn(L+u3uH@`~sEawz@u-SEp$l=+*qt7A9cUL(KcGJK&pr!sqI3T6}b<B&~acpsMkE`?^FWlt4coH=YM*i;I@3l}Fm)JUm5m5k-^HX3#$pqB-LV8}qC4w>sy5h!b0kJJ8&0tA@2ftYg<xDVZ7ETKjWp((OD1lNj)YK(ejFOL(tFWO)%KfIBo0qeaXa40F7LX*LGfuyJq}L3QJ@{M8Kn_V9EzCNcrdOS?7Yh=bb&lFja@SB~38jRiixTK(}1|0GPfD&OZH3=*&EZnur+F!HM|fQ$WsxP!4(evPTM<%<@PL9TXp6>WV9{MQ!9y*@DtQ<j=sf0w<RwDUXsFbA;RIfi9_`N=4_kIRDm$4b2_F3s4bycnl2YQ|YukQai=uPcg;!f3sirluy^EOgTsDA1}AwtkSiK|Rt-TdV24JNxq+jpjCFE2i<MagA7fUy`nCFt6(&ay<Oo@Ag0buabU-z0q-541>QjHeSQnubwuvm(ctyMcowpE%By=Jd|#yI(h4OYn1T73*Gytk)f3fRzKeQx%io!>c@t?_s#1%YI2zWO%9*SMd<4AHH1P*)k6ztyqt^PLw<kt2MkGIei+ETAMvZTmkjpsVOWfp78fjF!rR!|<^>n~nR>M`v0L8S`^Pj3j<$Gm=(<rm8PTuKXUSQvd@}?R8pW50Cwl^JKH#;Y^#`{H;S7uN02UU}(1RE?h>daP;Y@wKjdKavw-WXu`ok+D(o5XN-@+NR3A~cz8XTVIU3{oLwV{RmRZRsksOAOP2u}T{em^kt#=a+eONOR?MNlDqF&`d%4%*sde>5sg!xSQ3v`Jl5eSn<-0Doz(eMHs<HC4)bu$srOz7;Ra^Y!?R_qH=tA9xB{YG2A8?``?QJLX~FvEaM<xE8l=m1<DywsvPuPa|LB>72C~t&u&46%9T}o>0M$EBZ;-RjIiF%;C>Ibp1NobWVHq*7eu5di~EwmJGE-u}0Nr*VebV4@w9d0jyXZDaC7-O#nOcDAmk!7hIP>O--Az-fp{r(2RlFjIrCP%Kh6G)+~P^K6cbsHIi`SBIIeZ?e;lL%I3o^JJInkdNqmacft0>du!2VHcOoyGF6OH>&DD5d(Fv)FZXSBNx|k=oMrP>`#}3va4Y)@-CV25UHA;9YYB63@JBDAPVIpTLPP$wzMN36JqKhuW5v$O)VG2L9P<JV>et-7sX=4GM@Kl``G<|s{QaWV*lfke^HU-4VA^;MI#=GA;PRvEO8zgaBF|@vpu{Id;JL}}AJ;eUyv&2_a^td7Rm?9dkKP+l8a!>)zhKzl1D^58p%HvBPC2tTd>s)_rN_nr+1X9@*L|?327}sj(mrc*YQUvS<=dt%yZJDzDqKD1od(t?cgLPw182ihBCL-OP{F_7W!l%IKh@}dW|>EMciFBqY4ZuEr<8^NVmdY$!DDaQrjXj~aUpO!uQnQg9ms4t(NBNQN9XcHRx8&vmC@7Dc)#qw8Es-LyN6TxfP&U_4E*dT?*b*%1CUZYJ7CY&X7Eq??gnfsNN`SkYp2F}&>Za#f032%R?ZK=7$$C5U9YwF>ywK~nRmLHP4AWf3Cd<Ko?c?HlAwf$h0W9*Y#tcWF(4v}&Pc6dbP;8?gBucXQX$)n{sQ~H@ja4X=i3;&Z`FdXzVy5J1SEM?(@|!X6xf-zi;}GuBIoP9dOkeI;C7LU+ucW~{&5j*P8i?89*2GtHfOB8<rMu-?0`?LAl&;#m>-iot__<)oO51fxdkThC9?uV^&bFo{S(n$3J<)5pjm@?=yTksVEmGWCL)@;sv9|Ox-{%O3)DT6O-x@r?+ZpjV?%G{d5u)WT=ESlyVLaRJHI??FmM=YilQWH;K?BuzZ{c89;mLZylZ457f+UE3n4zNcI8jx?~RuAE6J-jA%<f@<VIsFMtsi!@bj>LZZH>Vy%^5j=KcL(Rx{riE{9m7hPZTcSj>`kU$}4MbBk2X`deU`t>-@NWgMte2IMIW?_P4HFG`xU+UEl*@_LA%O?uzImg}_=Y~m4zKBjN)-F4-Nu75c(f=QGFhmi-iMyr+(zv|*C-r?-^6VwJ@KJOlilvw^c{HX&~s>}?#w6?<>qR$P?C&v!{gvaU;JUIietDburYt#y_jqdW>dfCu!DK_nY`#WbjU*odP31R9LLdL8rqn><SCQfJogX^f(>jT;zd1$V7Ll1*gn76}`cz+yzb5!?ZM;%wXECLIYnxF4J--mGh?OGeO?q8KjwR)NIINYwC4}w=|&g0V~XAUt+%@3PZMW02gi%!j#EI27TL}^rENJv=^*ao#~wa6Kx<YEA!u;Sq}YnN7A(gNv9{l=Iw!ce7F-_!3cI>G&y^yc4?x(KFkKf_6<8xnud3r)RW_@(xj`Cre4e@i-1JA9)iG>yVUVs~w}moD3y2Ih9&ep+fK-B-w}w)@jjP<0p4d3s$Ve#h|U&f&ECp!lviIhy#Nq<#2F7jJXz;C`r=-6wCn%3^oQ%)ttprG3xs9R<Um*!=_%TMoU^KxOZ*W?nDu1>eBVtgu9*X}k=Q)Mn*wWoBR&Gho-{+`Eb2p*k3h!NuC64JmSa{m-TX9pFvBua0l4J&Eqv-SQ|){Xa-KZ*qM2#KHw1W{<VNUE*HPrEdps>fD<)bToNq*(c(;l~NL&e>TLWAUZHS>NYV^2D<NH<&K~6@fSzSRzvJjYp)t@-~(D=Lv(%j=JKo<Y=l{iO?t|{(daCv#mo6~u`R@-+O5;bRl<Ebb&rFKjiyd&58%!4>(8(N`;94hA9Jo0v3<Pv7kQ=wFRk@f^IkMbzR#!SLC#1aFfq9wbcot6Pjtm|5M~FKVp3Bkct%xvsM2TocMtS<-PzJFFa5Z`Zr-5lvsx<JbiHuK(*ZO-z#5Rb5tdH?=Qa#K?bXEl*{Re@;z!Uqo*&{BX(+W|$g2?z(o+L%c6avu*EwGfrTMO8A<5UUWpnvtvmVypXvcGK6w)`A0*@}Y>=w|#4#$t$XroM!4uOK&)q0-W5-!}BTxp2l!aR9&xIYj}%JZr5=%?+=aCoY}ae_Yox!7aYhOvH_p-FML5w^&AM2$SXNbtAL$OUM>=fE6m+7se}#6)w2b<d)Tc|mo3EO)!dXz7iV6Cl0@QL6U17r%;Bc>UXMOYyZ|iO6M{HD8y@jmvB^N5M$cLyY-MH|>|?=kTyMoqgl=`+`<JL8XClsBWs%v^9x^D4UKg$Q^c0gXbdd8DJwd1+f!2A_=rVNTBi}-rn{4PchwW%B2(^4*5>!USj>&86zXwT6Q1+JSNs?G3Ewy;Z|x-`?&gj)ibxRd`kW{Q=du6)?p`Q_UAK7;VQW3yszPwO|iGI-1cgj##-iD&rbB!p*jxsZSzBRBK-v>g|w`!Rd(aEh2H@<VGEd?Cv-!xf6V>g#Ep(VZ_~hJi5pji*(%4!{_59zDjB#|Z<&t(49KkdGD7MC(ne1Gd0_al;ExUv6WoAq|2xu$cn*E7t+C|2XYfqq<i36UCT+A>JR!Xfw(Sf3cPQg0YgaeEl8^^v2`#wAw7M2*v#*<;TL9NX;!4H3u<QdLB+fL7tii56eoSl|sqV%-pO|!}16c;utb1jQL^?zMhNe0`fx~D&{WD7BUF!@BiyP(I`qa7at7`?KoP8H@=;M3J>-D{~Z=uml^?Sr=xzt18)vfWJ_PeT0RkHpldz%#l?D<<hIL(sRWKI|R?5v9YgYlXDOq!YXBzQO!Ll{<Hy_e5)`lq32Gzpu($ukd2wzBYG8zhs-L}1zS9;r5*`ml51$r)&smsARSQtNSeWTOoc9FEfs<Z(6ec|2Xhg<cbnMWX42Eq2XC6}aAMZ6t|e);|I~etp~D7k8h%(Ee`ISg;fc@L(r*n_O#6ACU!jUSh4=e!fiS7~is|{foNcpvr__Uh3IYf*GT?x?8GWlV4=3-91hB1<%}m-x|)*b1UmmqZvN!;eOnMuyOf<_Zx4%WK<2`w;^fqUU=2ZM=Wim%eh#d4(n{PdD{d%%e=WO6N`><zy78}8+P*^>B?u=+;WwwkSw;|qis#G=SA39ODAc8#~*<$&<@2N&dEU~vxgoe_En|GdNBDjVI)1>{Kw}pwqAi|y@XYScQY@tXK^-O&$h<Pzn87=AGbVyRK$)drqo(=z%Zhta!QdGxcwzMQW_P%jW6Nq@zMjRzHZipP6~f+^5&&~7cT&XHkU0|q*u&}YsgHp>-L2F17>#LIt{+V!#paP1<<)vICfh7%};2(v2r#bwb4#L>`2%_sy^Fr@yr|Z?};|nh=A-UE0u&Ik`WJMI+y4B7F~{wxZMqt(a<DzRP3jfwnd68ef^W7c|JVMcp#XzR~1;H56c2x9D4789t@MHpQbgvRi1?npkAwoFupchNXcJz4VEVJJ~9^?wZlfI3}iO`Zk5>xTg#VWNA#C|BZ{QBXU|o2UD<Rhmz2P}<Lm4_ZUo@q-8@l}E{DBme3Q-Ze%}&%!ZYf9_fSR<=tfnujy$`gr*%9`dmJ$kneJjV3oH8vzC0c0lhx$;d(100XIhCHC>7%+w`q_?@1Qn-fjnV)<Me?OqYlPTn$w3f)kFdDeAJ~!yI+23Oi|KlCzCNfUcZ)&!8rx^x1R5%-X=A1^>SKx3rHm6*<-l8XUj%ut?<ERx9V})Bh-#Fq9crj`)*tp0JjKc`}MuU<IQxY$Y}S4&E?t|f1dei9jgqo`49Y&J3HEuJE1e<uf%hvt)=xCn;bq{%et|w+qFA;X=c*BVg8q%y83ZIYDtEF0Ik}pzxFrSJH<mD1`~DHvMA*3ikf@H$A{P$8j@4C!q=_;$UrL@lj=yDJlwkdlU9pez0VJv;V>E!Q(bW9e@+T_4#@r>QWmCGJDNE!&CM!W^i$FwjrgRH!3+><?_Ad9{h%&To!ySGk?djY+sZnig2r&S<GZ5-ADrHmOZBlvyF8cb!E+DDz};CSWZOL`y`N;sLF1O4M9Dw9!fU}{##wj(Wnb)UF~P><^RjcEOru#%BGs-onLGxvXO2qPvhjY?dgO-DrO4hz`lLEI!}{6jB-&wj^E_N#2=pVsPv--KHY2!o;4<z|N7|`B)~!PuL}qHgL3r_5um`Y=r+Y|=nvY}hpy?dYlqJ@j)NPl|r~^nsll!i+UIzy^-iyO^ecSaiqk`7Te8rh=|4(l5`l5GaCcuPP09mU+nJlp4arX9p<>McHSt^eYY+p}GdDiFi^Wj^CCR6*>#UJt0E>!@VR>kY?yKRs>T5PqI!!lN|I#x&Qc}gBM*<b5R$MAyE5IT2$xB*7Nn@EP_+%K|)M-NG)AH6n%%%}BJ<s;zcsi6l2G-8M`Y|C17+Aut6^=p$~cc)g}M!a?@p|ZKzEwi~+Z?7b4p{M!GJEA?A%(@Im&7RYHd>-a@*T)yh^>sbAy0~a)&~BPtZmq|@yd3y-yV)=LfS3@VR-E06t_YhUkuD)Z<7=$~1PHD?73Y2L4-i!2ckEAd%Ye_uw(tQipszFI=fTMnq=#*as_mcgo$nO0=1$u2fIN!UC9BOY7Pbi;HJY?Pz(ewe{>(tgmWX{?2Xh$Rc;Z@Fe#ud#x338Ce)GHSI*+t@s^_SaeFw)i*{xpBb<k|}ZsWV#9DPyBu&-nG&!eiZI(gpK&CS=uP&MwUTCIZtS8-xM)t2oE@<jh$*R<Zh!mQQ*wRoz+Od-+kZ(sXH5|+noe}9@wYj?;5DU!}-&m?Nq+GJ#&bUdpmbvxYRU>~CZ0iEo2vI-FA!R^s_Hg}6cd~fSNFcqd-BwZ}%?Ynh5ErFOGG*fUYh3u}n;aAL3p$+xc#<Ves;lE^jc?#>$gC3V|(5@n$eo;nD{hjhpD*Wis1BPdVM04D7rQ;SVw<nM;jNqiU{x^YKIvAC7S#4l%l6Tv$O7Z)_zRzAR^J^1FKoGxkba1JTKU=!VHdi-mlvRQ^nS%Bb`{-cqiY3<?y|1bKgxsIpJ|y2!FxU$-bvvxCIm^<oo+9rb+qC+0s`;t2UJdeahh*Y{IXzyAGsEnnlP9iGDpxv&VUy?%c~Fo%K5`R!GRe|g7AWnxxqNu9irY^sEP&S{gs-{PZWG5_$GV9kGqZeueL8@z$MuDSUWe+FFh1O00nO?Du~zzXSe;%~m#GeZFt7w1s~TQ_71~F~gyJt6`DM`sk(yy1y!&>4a1`dJ#5w$iij1ZHQUl{#9I<eq`lAkg(@ohj)4BEe?3I~mo}Blx=J$<<ibU_JJX|;TW7==@;cl&U_7O5?za@F3#o6iiT@Pfi+jEsJ3ov)}+!!||`0c4G?!doDzpqT{;dErim;O6q=Gm*;XbXZ)ckq$U_^tb??Q2$Wd@TRU^1z?m{8jua*U=#bliJjiwr>CaB8nF<XTH-Tr!?agVuLEgEK1(*tI!Xa{dq+DX-3?!ADutOlbE3clym#Jde66dn!akXIyn&+PhKa{vrxoAmP+{jRo?pm;w;+BX?Wx9@><M+s<YW{@^J-wMF*lyfX=V!-1%n6TYBwFH|pnC+50L$NB^8XaL3wadK|N}j-s?Ne71+Q8LI_()Lt>$SJQGP)gDm?#5K0hn#3(%$bR}rxxU9D?RD+@T8<g5F3%brvbR8bRtd72Q5aRq>ZQE}a4_S~w<6t`1M!}=h<8}8_!xnWKlJ-|y0OT+XHXTYx1N^IeM;>*d_L_hw>x|t#w|QD7wb*)b=T--8TQ#02o73FO3gNeb^OztyfGEe6=1jPjGcz<yywer?aTfl0bJB}xm2K?wmD-?C^D}j@q2DR6h29qR@~|s`Lf~+RyF6HDVxJ&`-iWaq)nXMCDnSA`nbyA=6?rLn}{nEDiDAMx99#W#lDD547eecu5$VvE19`=A|mK|{h~TdAW0ebk?h)TZh_I-^`(lRHU5(I@L*#U6@j1MM-_nkGM|z?u(6NS!zk}RKToDMT%76Hx?Mk=pf#SchT3(l?e|Jgt3*Bz`ysR6Y(7YmF)JOto`$tS`}e#>O<%4ot^Qh|tG&m=%BLH;6-|S~R9T61t&?&Z2bt`p!gr3?74sJgwwIxhC>!}K{tit4A?zO}(ie8z?M4paE?()(0T+%S1nV|iDseONTTJBj(pRd@Xg_`vHWWU*bL|Ph(jq+-dwPkh`s&(*^oBz=fLShsFW3MuBPXw$<>q|Z3~2UK4QrGEw-5UAPxjmvIMlbKY1DP$^#f$EmQJWmMwrJ<!5*qQI;tMs2|2jEk?jq1=O=M{?B8x91?!%MTr5Sc>@P5MM+FO3A!j{WKXs~cP%u?<e$Q@&xnbTOyY<%$^30O`c|<LX&Ns_u%(pMt#TgMto1f%nz1xj3u1#=I^m4rX^dO<OU04!)svZl%xvs<R{7YS}ZuktDS5wTx<jU@U;MEiMJA-(eUvhm2j@edgL<?lRWX|s!4GP$_gLf}WXg4-VlKZu#qW!ijUw>9Rtp6$r&7@}z8P_xZ*CUrfE|HsArS3iO8(6<z-XOUfY&}`u>|#x1gsafFudF|8wB%;%f!$D^D<SUj(<W#(L`V5j^t}OO`y$n@BumPvS4MvCY2W`ypt#uS=gw3+c{`&crj<2~*`l>yX;N?}CvS~>_awF&?W<a^_sp$Y^o~bD%%|7oa0I0qTYaAa{(Jt7X4()<>se>`A%^U{-Fepft>mE|hFQX8F-)x2EOxF*=Oo|W{e6;eP=dhQ{p5XHgF)1PHT#=jLLrMZf7gxc>3Hn1;3Tdp>=5sDyrs2WKQEzswr`CCImNbz?5eeXP)BLMD+>fZBux2NpWoA7_c6SBw~KfW<=FhggFTy{yuj59K(@1!L$fs_*wq-Z6&RsvqshDZwxysLl7_g`mCItj+@F`GRTgic6vL~)DeoKfsQKBAE;{w0vehLp&RBhOVhv<IE?B6eHo^TLjZBQ_#l>k8?~4il4D8<n6MeV#A(rrgpc?Z*v(?mS8VBV0jq#6(8}uAQJD82V)}34RY~J1Oqx(8vTKrXxjC&_vVjP&>xrXz{U;J_@N6!)5whN|xwg-dpHqxRC!(9o+ZM4QLFvOUXOz5Yv;``TDMBzo0MjT5y@SySPq^)3DSgNV1I05Vq=h_at?IunHSG8$-Z+U<cua?Le`b}ZvFJ#>CDlxBof~+frZf{ck?Kq*iVT~=C{;gC#hK@=JbM<SiKWQuUPP?E2rKsa|FVkzt^|*P+Ks|0&{|dSC+cc=*Ib7FFXqX!lnc^Qs-dZ0P?WA>SKE$C@HRN09-`{W>JH1Qf<%AdQflnIX_UpMX7M|9mF48I&pM(DO>j;wd3(mYK?+RF$()2iO*)6=$QV?I$>8c6c4%4h^?ArRIt+jj4Mz-<PaF3b(bBd0#tNIjHz+o?vldTKIS40wLPe$t4^<e!Fmaq6eg26S%`bGvPV7@z}|2K{@<zB4Cyfdh*aSmJU*+&hk%m-~w)i-A5(K=lE)0YvH>0K+TaxniA74b3Vemu@|3*d)1)LXv|I@oYNE&Rw|`1MZhPI2Rqq;k70h#Et?jCkbSHj&Nm9M}`E<*Or$(JJ`i<J#c!mr&o2Z73FXyxMeECE)r;L6?x04oBW7M&<l41v09ivs~BrRh;ojeosrV6cfz2J|7%{cMrAjCsE}!mrp5aye4ixXOBHCLOI__{l0AIihapoxXJ(J2jhB;H!mR$uAom+>|c6$e-Pj4EMuC6z1o%>ZnTyZMZM?HzzOSDe>OM$Cv&T%-Qy0vOR&m3T6b)G|B0hs>C+40oxVfckh+;!y2Lh=q+FPxOB)!)ig$)VrGB8eU32XDFnVf$cOUeV4B4gEnQifQ{<g+fSMy!KXOQFSeO&$qLF9!`?=%*S_N7`++5^r?PI=MId71S2#W<|b@?4vk@=P7J4&ff?vyxPWSM*^ey2KuOp6s`n_u5-~c_)BIN&$Ugq%SIO$aIJm-`}pzH-Y%0!`~I;?%sNqG_93d@-<ZMY1dVAQQN!S3bbURDe{THt?={8#^!TbPvx?=7(w}i|9mAsT6_@7tSqc-bm?tHc>Qf=e}#AoN<eJ1Jr><|w(-j)3LkxL6}i3c6B}}&SRycKlgC&2X<67=&eB=&-PKkr51N0+!up+f6V$s><y;q8M}9s6*YO97C?W>WJJ$QNdhEnHwE|II@%jNozZY4Wm;PLsViox*E|KB;bWe0QJ9j(ON}uZ6MG8Iu<L(L=q3H5PpTG1Ip`WMRcRV^Kq_<)okd;=-X(r>Cj{qJQ%PS=Sgqk6YD5YvTPThhHsWsBO16u#@m$ZU5l`k63FIQ%8>N@hc@5wK3AKJ6Lavuw=<06o5Mjc;&3rkm3nX6$!`_RuOok~HrS6%W*9!7)jjI(wd7Q*#g^?AR6>cx3Nc|7N}2ZijkwtIg!?;AJikJOr*l?^jCyyhrdo~%)^A(0F6Hc&A8G863z)hb=?(V_!h?Rnm?#J5X4lrhn`JRiw0JumaaVRu|jpSOiy0l|h}Z(sTVT7gt`{o7K-7@ifikgtkIfAk&GyUr~L2itnRKU4nTea}%+@<;Y0fd47O!9FnTd}ay9%W-o2-p&sMylR7_R`&KK)Ym2gGIhmyNLbkSSglE&ad7WQ+p(}}9;qX>wjC|Y61m;Uk3khs&VDZg!%cPIgy`coxMuxs^<BQ;;zQws8_)~pP(%`|bIW%bbONaEqD;lts#2k>O}$oo4C<&kIn-mO3)CF=rGnwzZs*BwMNAj&u$usMo6^o|-n+G{kPOLF`s&vD*ag~>8Hzc`7SAe?_@`y*wYHg4)0??*>k_J18U(1>WILX(rq*2@VbDlOT~=OwH%)~yr<2WVAF=gY$0CXrvcRfKXv6FfjvSbVyX#)}3uS+%R>10VZf%pKGHdLna`2ZD27Gv@$D@$_ex@Ge&6UrHSC2{ac6e;{mAVa`%k(lf>Gn8Tb|gj)Pg8CGPTP+-Cz5@7G8}dE)!x#%H9c{wYjsm!9wPtkaqWKAmkmgqY=-*-H(hehRc}}|hd%m$&qG~Cee*Uz8kF@HG7T;wb0qpBjdkvh3jaqHSQ<%>MI}(C>^Y3K@ht1a<!~gUSoq{N^;oLknayf2(}ITB2_1LJ9$4#ye*Q@L%9)GXO?v$nkA*0!03r?2+2}eoPFE-FkdeOcV+-LjDxXkuBp!hv*k^U}P&tnllh=e=*Pco7?oBO$Mfe_UcxD|~1-f-CisHCGEKcwMQzm0krIXWgeeh(amH_p5AM?fTIRo>Q+rb?z7A0I0=KW0<kf+0udaTmz5!oD`xudVV$!>SqwjYtSU|`ldw(Lflu3M?_q=GQUem!<F0%RuWT>I5pcd1pcl*|?@ZFH2ZVtR0aTnk@cXY1(s(G{W*<SldfbWvV^VR$S;wXVNAiS=9abh)fE!{jip!!09c`Ie&t+en53XHZeF)_wqgz+apHnk%F_Q)r3XyxTI`{6M4l74+ul15q9Qgcj#|2Ngk9oXBo_aR+2Rp%wyrbFGa(25%(a=5LX7Ihet2p5ENU?>eglhLrT_;zrb!7Buw(ZnS)yIxRG(syF6AklRNE8|=<Wwxa@<HGqB7*D`%QYfYvZTKU}Bq?BI`(ec6jwp}2*>TW3}F)7E|PPoCDA_zlEp%}L1egF_Ox+fQE$E(`_w&qypbcXwKBumD5X0l#anw+q+<!^yS@2nq`%U^d^6SvWi1@#?t7TDxSWB%@NuP(w!kR73_FuTD1a$FPft76@~v`zc@o{mxlD7!nuJS)_Ihw9+wNVL~ur&iH`ZPcw8B{H5ADUZ$TRY|R_8bW<FJ*oFzFo=d9g=AWvW!O5ElgJsa=r2*e(_^~7H3l{{uH_5<L1dg8ACc|1qZ&d?j;vId?x`An6rn??8|uy#+n)yT2j|{emq44z$#c|Z|5C%80nf(z!VlTP+s%Ue=~HjQCI>Nx8}tp6vA*iP&YK4m6PD(U`kzEE?g>3@!<C+eZ6vEAb&reJUJ0j64-pT8=Ju40-9=)?X25PopI_r)Tb2U+arcE@e+(Ql9&JD)USwlJO^kh8wiY`%qT_n21+FU2LLXk8&dQcq#Y#wmK)(4c_-M1c`;b03I<BjWY`up&*!Oo^2fqMQr^HJebhY;yHgqzO#_r^vZUBO<L#u3zivWZS&p&*r|J|deCA;2^SxaKST)%netVfrnPfx*nKKq=yo6%#?J3%q%Kapx>*Jz)d^>d|^EPNb)h@w#@EOjjpABWIqr|&dd@>Ty~EM#&MoY4FA2V+j6trOpWerCq?ox)f*VBfFzE3NH6#S(LDdmcB&t$E}xr9)#Tcc3%!sbY=1ih3?~h9|u`xteJ2QfuC?+G#Uv(Gzr)nV<c=-C@C?KOecrc6UVN-QnLW{c$ArjXkE|k4>FeMfK^LZS~wQ?<A9+G)H#Fd=&hehr{!+`qsbxxE{QkU%0RGE#I#YMZ1k*AKUyg&8WKMOOsx@QSR)!?J2G(l|M@NFXR2NBC3fUYc{(Jwn=)C*PPfHnp8~5XKjSPH~A*OhQHN&T)Ff}sJgy5ObMQ&lNByGz47)NDVVdjFDl;K#&*EI^QAwjfCGfB-oEGg6$zC|OKWuJl}SPXs9&bZ2xMtsk$y;V?=IqSWKL?jorK`O&BpaJNyg`6ZC{?FYB-!kb>;hR*~h)ZmWZ{Pg3vh%Zs7U?pjz8=He)V7Dpl+<V&C_1-SD6HsIQ-o<$RGKm+00#3-~8%{!wVBJrE#_<!bltsP-AIVTay4E&djo%7TbsIVg{0=lNwYmYQU*hoJqM>OcWCbWJcEzRVk&ar5eL5k&0}D}k)8oWq1nG44EmG={&Pv?g0stAb8TppW|{b`qQjx-U*tMIV!R(Ozd;<3imXAZwl#c0{EaxSNRxpg%2Zwqel9pBeu37oAj{VNC^8p}jP@@fh5?_qJpdow8$(>#ihLDz)QhXUggU-fq^gd(0j&ZAd|OZtmmT0|xobsy;Yn2&IPc*$S2eb+?BUmils9R@wl*UKZ#|Q2VXQd-@a(S#5=mX^pKK#bdV<u&a7)lB|2(V{_u1lajV@p?lfwzUE<Ys_>-0Q@1<6FEmAYGGAvMz7#)6waQi&C#hfehL@Lmx<Z$zJMV4frnY*Soh#*G*VVl6fO;yvZxuViJM%O28UO__O*lRsDJzLz?~%%PennCGJ{~f5vjtq{joD*64zZ?Dx2yHhWXbFm?d{{qOjhJXBN=c~e-G~LXDS9bp9w1EclLuT-8BeBqosOUnqiMgA{rFBNUzq|4z|_zbsXM?T<@W+AOcz==v-}rlUIsY@_Hoow$zz>ud2EFmfdODQ|nRJ?qu%(Og`u|LGb>i5Xrf6eO~sozk{O!%kxk^I$NVQ|Bgx#SWG%(i(YYM|MnY(X~m@H!0RGa3!4S+*mW$`#(=)OyC<r5?)J|O2boQ=$u0@W+k8t^%J0IwF<Xwg&K~^~F)3`fylvQBHu=KmN^!m20|=T-1eY4Ot}%6_{}ifu7h8*IacHO+sjqw-ts0NmP|Ty8R^9S+y?4mVP+0whM`lEi4&9sSNq9Q?!fkXP@6j^{{5D@~ACIp;F+}@b>Gh;@=RNr4Vf8&*);ZyB5%ctvzxL2$4HaF|nf9tnOju&I{tG|vm--tzNgYNccmb|U_~FFL!W$D=6&*13>#Gjdj?cz6$6ov6Y%pfJO>b9+>S(zdq~mF_adY+4G``=7a!ZKeACWKTihAa?Ug>tPE0Jer;rY?L=w8P5^Adgq&tte;*&sC48=q^0*VBH7J9AiX2~G{<8`$wsTWQ|lRK89;r+%O9&ZHl>U+~12Gtf!x7WAmC!1HB8M#rh`3@>gCOoyxAaZ<jN;g~@d<J2yXCkTX8sWu<=b~ug%t;!y4V_{pTevh%_Vx?_B9I&<b{pPhz^qVcA|Cr_UP^~p;;eNvnZ|j|s;lJ)M4B$@tn({6KU&6dJgzsae93dL$ebC=<JT<RofgLFtnOv3S=TN@GbjOTZ6xnSYb>Oi!2Ss1vR^IU5YHYq!W?OyLQyndb<JR=J*iJT>JsC&(A8~h=-D?|<ApEw5JdjeGEz9Xv?^2%<1Z?*i9WN60(sp}mi0)sv!h^DHdb<xVqrq58cql$r-R(nION=zvhUjwqJ6_n4uhr>eL%zR_s#NzDYFh!!@nJUXj@*eaa8;;r?lh^`a`@e2_-@C8r#n_&w~BDO8v4S~U-$Jj(Xw6gSEd!#VPJbO$bZ|ew(=S=VZ0dK@xrIzcgUeKRPp#Y2;7dL#I`tTNT1JsOJB3%G=US%-P1Mbmh@5dF4`!RoL%KHIrnB#p+NOuh@D!#y2(h{Zm*i}%OP~{<M#P}X?oG_;Fe#p!<JH`)|us+nIS!TZDFnEkUfM&|9Us>8Y#Q4kGW_(=FdK}B=#a%=eajIUSFj4cz$aXQJU)a$Ttqpe1d<A_#)$4Q?*`1Sv^lp(gFALCnBM>CxvY^_#RoC54d!hRH%ef0Lbn;+9dE_l4=jzfuXI?cdhQt9T<&FZM~)E3+r`YAEWx+MFQ_KYzDroxl9nJ(Mg{en}lga&FSDcN#iM?ZuaX@{d4m=YpqjOE?ze}IhS46eZ1v_OV`K2?@8J_S}08>hT5GMF_H;(5HFw(t}SdcN$?j$j^@vSeoW}KMAdHVt(?E}Pph1S7+N0Quw_VaWZKaygP&%sA?8os*yB43?9Tck3G!DBz}5R%1<T!M{665Va-eWif^3hXN-!MV&`_L_(IVYiCkt;fD>SvMS;ct<JL1;v*Cw4v`L@WPAA!S4d+fD&>`fuq@kh~t4s78!ds5$8{|d(`V$JSkJej`2*!Z>*v=ZMlH_JDMtM{z-ZiO3-R%_yOZC0W#w$fBOuh1W&az6xvj@U(+=jolED6%Zl&L;^wO4-#ve*s%epMOks2;zNo1X@xy3{TMlR-c!hyevA_wR#}+=;<%4@o_IwBUPlTv>F1p1_&;FopnTeq2+@$>Q7Aa^1s24Bk2!P-Qx(P2Tux5rr86rx<f+yCYK@nv>a-E;O29cNpp2uCFz)1+fWl3#xz*vnxJ&v-3MER?gi3pbsBE3b$JNZilM#xThy@`-JHR+8P%Lh@g(chf(xzSxF>V=YVX>XtA2jFcXWHz|E|o2ISjN7W%z&-yQo}h_9a^p9t@lSi@F;S&y~deu;tN0^g79<&eRt=uhj#4i1k6HxqF*xd$*mcULz@V$#2r#IO)E>1{i;Tc5?+sZ=$cz9+PJ%mp3@SCm!xJka2>jrkh0Qq_Zx{Z+5YRH1%n_>TDEv;!~>EjQ0yGZWm|pI0S-XN%ao?eB{YEPH;uSVq>+?q39aAr%5)uue!)SsZG~2x!3YK{~_)@yBs&JgyHY{70qT?f}zD6h6HoYIn5b@APJHnL4X99$?hYsEm@Y8ylcxra*%D=TH6Ztu5DTI$2`OCnNQDOc&h-go9t$fWS@P`tD|WWg{oV(Zr!@6sxD)-StQY4p;W0@I(H+$hvC{i3e9C)PFgN2&e!gsWwHrmk(z{QE$j91v`t${6xc`OCKnu*BbNRe)p<K~F&v@LFd50Nz3i%)>SqTdq~wk2Dmqqs-U4<I(?a~|tTl|VnPt0X)wX59@nEqE<%_LQeWrKDSK8hvVrie3B1pZ|nMelgyjN*TQ#ykou?RH>X(d^R^v=x&>3Yc;eC1wy7%#TlZ3}G<MmmXQPJtW3tHC@tS}e7Vg>s`F8i;;-x*EsWOiH$?cQ)B!JTPOUomHLPWLOrLR?pSvjG>e2R^#njq3@(xc6+<;3^e3bB#o9Wb^B;F>z)^8#ze^FG1on=Ceq3fWiQjg-51%4AEN=MY4!2aus*Lwa{)BG)az`mAdzvH^5MY1N-cdI1Q?1PwF$-MmDO@k17?XTqO1+pZiO>riBuVrA-_pntW3GEqZQN5V?$dZgIl9WU#q5bU6Z%4Nt7IPtL?~Nm}7R|TDx4Xb=8_IZ&`KazCe8%Zw4(E8ZWK&t>m=otHiQOeAHi+SCd44ZHwYAtCtQKi9S8aL}_F?^(EJ3Bx}(R;$1SB56<akq%d?Fv6;7#%s13bDk&Sdf|2YwWwfb^bEy_9cAU0|uX^rFU#M%kjjcCyv>D7)l6Uo)a=&X!r$Ts(iPR`pBb!@H78&MTx905jY^m&I6qTN7wvu-gDSLGtR{8u*ijY=x5_~1o>wW2U+MCcz4OU~L8EEA5q=j>-bgPWn9@AIiV|kpwfB_vkZ(?pzF;^D@=D>)KvhtZSdF~^_l#TO^!Y;B{FohxoUp?os`)eC#E;FT5DM{I3k7gT4+M}z~($S#Px9}CIY6`KQN2XruhENvD_`t;uO88o-EJQ}b^fH-2P=)*g_v4&VJ<OQq@ydMDM2ScuAIQ)B>2rJAVe`oh&2rbBm%A78K_ox*_@_&KjBRNbdAW~m&+^S=J7eqF7Xgmbk%_n>x9aDqu_s9k@}u6Gp<=nfY)F^OW&_dkDtsHM%ed?q2dLbLIWB4po41GR+L6z0VBB@api=m=naPE5=3LVuN7$*b)@7EaJ?ZcO>Flug{CRme8?<zqNi@`KpjLTDJFQ_(r%S%d#kmVhxQLu5;%57(JF_(u<{+g?Xgl-ppjPyiXrvWpXB$T1YGx~FIX&~JSeb9-TVT-<&B%#!2fok+22!6<8}rkJK(6kJ1GlCo85oM4dUhHNQx|ndj4HZXhGbP&nD&-2d$7%H?6H8>lq6a%tuz_ztRs2_uzc~_)NNk)q;Uq7Te`+YAXZs$ay?f!S7l6s)CY<sS!y~b$kc`m6gC4!t(=5PuEn}Rb^T?t--#3BP14vt=N8db+m*8rq(fcirYKqt$D_-EY$4N{N>tn)HW%kPd96RHk5lYSKX0}hF3J-y###w>yh>v^LnzB~QjcHO&r0fqTREC2`>_F@9gXs}La3DrbDpYdwB%B~n7W{!S1+V8=fFY822D$XD;r2-X5rCEN0Ta-R3ZpRg^iwbQ$Ty+Fd>;T!p34Ii7|XWn#vX9c+g)#b)&UgM?_na<|>wUHI~I8w&99MN^Oy3h8@*;Ql)Z`Ia-z=%f3e17^+mFa=_NEl*5tIE!TnJg~XN^se5gO$I~Z@jYYULF;t5eX{j}~?Ad&jJklLWRQh83LON>bW3Hwz?Te~&S$A_C*DkZPhZ$K^lyW&5t~SQnSnErXOrgd^<`zosi#QzWRnRf-I=z{4RFNEN%|5k#RK~1B$1*V-B+{+bMR7H2b>+C%mCCf(k;<xUSF6<}GmKuyyH$0Z=@{b?%P5#OUgRVjM|@(Uj7@vr<qyTW-I98vu2;I<xp|HFOioqN%8)BxNopQBB8h}~a4t=`7Ikk<n=y?XxvpV>n<+A%SDO1%FG{KAwsOknxsZ5U)$x3FZgd9S!<j!}&M1glaPDE`<HnFJq7#qYK~H6CX~-8*F9HrACR;f<?k=l}8>QAm*Q8C-?&BJbi9B<Do=<P6X5XvRDhAg2Bx8+@>UEETDI&_os^guuORIBI>qIKvOt>c3A+xw6OBd^XOFM0w8mpUljM6W4YuhR<cja`tQLj{$G()~R>Pb=cdgh{%$`6d=3&%>`P1w4lfMMOS=Eozfr5(mB&U$9LFtis6`@+=-I30~?HWy6VBUC9b8MHRpMM1mrZW=yRWt}=uMMYu`FWt3@y6#jB*JSddij`RDsu89(&Q4hyp2o*9Uu<SS*IxuGGv87<EA~{iD5Xy(nACaC8fsfO-`dkqsMEQMRcZF;6(LVcW=A@}8>Z#Px-3Ka=uJ?z$(bcBc~&+hkU|2Z@U@#NN=8I?>5=0Zth<)lP<q{y4>)B!!A*M}Vj|-z4X2fkb^?QO#MrHG!u4s%S}kPTbkG*b8JDVNEf<JuXT`3!Dec%iW+D)DhL@A35yxUPMi)^Osy1u*V$-h$Yi6lpsti^~K}*<>Q&=`}Uoy|8YEfF>Zeq&iwCJA)daG)!ZLiPW0R!IC8732RX(3URUB0y3Y@Jm%WmQ@d=*ZMlRj@Ltb%vWty{3p&f_)_H%%{3Oe?Mwr{Cz|=$-268CENG+mL8wnW^Aj7*qF=#r)#OSHYtb6rd@5i6PvEmQ(kauIi63+5_B|*mO|%~aZo>%v|603;c~56Y01@WEK_o=zu8Mp&3Rc-+pZcFHJP-6RxqqF@CQe|{DN%H+4BZ@zDPIo=d%kb)s!Hs1m$jot@&nL(df4^|E$&1ucY<SCX*QSm2H1AbzW|FYq~yC)Y_+Qthdn>idn|g8@luUdc$>L%uQ4JWr<eH)QM?iMK;4B?_6h>58MQqrRQC<S*K7lqhO{*45>!KqU_08#M2&6`cuEo>qLo6Y(-_9D~2^zL(yeDQb^lO^iZNuRy@r<COuEPJFdWh(G%^Gr&Sy#)|r|T>(@O>E7p+#?HUP0o$a+E>h+B0v*nyNlsNQ4!4;{R&(ZM*$b5jck^@I+97XHl<gywZcBX0nB+K?g>gI43Z|R~-S5uqLoEw~RX-S)3rb-jXeCb9)D(`SHn%mVaqb`#TU>Mh|tCeBIJ?`hL8I`1=wd&oqu|wAE;*O;W8@qh;Trx#6OXHZ;O)ij9Rax^B^(m?iv8dUpjz+txY^md9nY3IN?aKyvr{Co?t(^hiteUJT2EjtwVOi*qzEx$T=4He&v|f~QeNNwAIm29ZZI0R%<uQI9t|YDPHLlA~H%Vuy+Q?cfse~Jsm{biew82R;Ql6mUks>r4WOG2|1q-eUsrI@!TXSQ!wet)4Fw}LMn4F<EUaoweez;@s)&erUqoGa7-Cb3`=^1y?p)AsA&7FEQQbQHtyj~^Aj@oPz9R#{Vb0ppM<Xp)>Ww}^j{z}Hqghu3O><m|XA&JMPk45J$Tq)yJ-h8~-bDle06?LsQAq)0lSAq^a>qe%Gca5=PULy4bomJcb)QTxIPCF*6VIy8?H0QKam#Kx$rRjjj(VVC1vfMf(nLDjBEh$e`SM;<N8~VBte{59=L?n5=)FKHs6g8K1ot($!E`*M$(2dcUb4-`D^@W}5yI7~ykTVdSh*9s%Y&KEWL?B_8#ORT_7m1eC5KOZRd(J60r%J1fyi1awn}WF>&Mq6FL^#Jdl$dT~B9Yl*-D!6u27hvC?={BGOvo3H$P5Oh(-Dh}ty=3U+MOh$RZ`{*ndYi2(ws>QQt5CZ&yS~c6p<QHUDq4x=BP~B5zMN~i%RRFZ`ZA>vgPPpm8Xp>(pHw360)X3M>|6q-=bjG+GFS5b-H27So0<fUx&lCbxq%sV57P(;d3hKbIS+@+wdk5^3z81B$Dkca!u8|V^<~O`7$*R&HA|^SC{$Y$#Oa2M63h+LangV0V3X|`_5ptTj`LQ;iep~PbBu`Smu>xlK6129Jb`nx^?azBm$n)EFMz(#%|@BbsAe}TcOW$#L!?|n+GNto{g7itUYzm;dyaHG$)IAQtPV^Wl|Y8^i=B6oMeGZTYc`L8&6H`=SxCfZYgw5(i|JQDbk-D=Un0Sz`MRME4or2Sowyuabk2jIEk{WZ93Wxoj-X|M<Pj`RT8TawF2EzUbGZtCPs1og3hUQ8p>uLM;NM(vK?8aU9I#~r7g%Sb~in9)i*H}WAu8XEbiP8>ae<G<TB1N8u#l<t1_}?TB?#+zL0Uca$DvJaI5Z!9#)iE+E)%#Tj7f#q8l%h9agtqyK{1t;-Zq>M5oG-yyr4YBi3fzl9TpYT0$10CjM@VP^1g_nrEsl)6}A^FN~uDNo*y(=*1+gGf<!w4OP5MWplMi*yYdbbq$sNA|$I(8`90_<w3pKy-cJDr_syB>#o$qGVAoRLH)8?oiNtejA*K2rkHWPNv@j25XFn?gncr14hsIjB%fMylG%nXXt6}Ws9tmt)zn?Gb+|G&n@W4^`D9X<N=c$kO*XzlqFl^TLEl8*iMPpfNg&@~t&KpdtljvQae5g~AX;s5W{c`lF0V3dZ_-O!q;XM<CtA%#N#8hk%2Gu8oNBGgu85(rVd|WVO@x!$i%@9eX~ey@JWYWK!6NYzp=d3l)Fuh+CSTu|!eX}is#x)_aAikDC_JXc2*<chTjw0(a<_^0mGX#+LvoF%M20Url{O)fRr3qX<ro-j)}RYfP3)bBI}!^;t7xO%>X7YLePOT7k*+(q@U&uyKH?4_#AIIW<3qyQ34|5JGQ5S|mWQyMGO=7mq{IeWnES(Gh0P`EGx@R|&6bC*u+k@UG!T^{*eq9*S_kc0)i>rf1^k?B07Y`6v^L)D+AV`LWd(>&(&|!cFt&Qo06CkuN@I3Wm}E>UWehn-*2;2E*&O>*89m~gyE8SatOB-`RME`J7Cs~u){SNK1+Zm(ct=J#jPvZIXjI81nBCy^Ny0%p64j>sD&0^<$LS`}w}!P1H;snvc_J*^ESP>kieMRh<iyZbN1M>bE@akDus;eq4Wo>ImI*bOUagS!`bK^?txly{=Pvo$G{C2_u_L{f*4^nvWJoejv&>dXcM|L5T0LU>7JbG#ljfBbQ`%5j%yjxz4>R<P&F(NwH2fD<$<i5jYKc*pw$ep^HJw>2B(koop&NDaaH+p$tUdLzpl(^2SuHoUaNV{2!n;CUp}1$zTv@1=LyPG%?!@_Q<)Xs<z=9)U-uk4@t<xqeYm|Ezp;cQexkxpGq%z9HV%ao9gxAB0O-`1u`bakPJE)Far*jZ`vPmvfHO|xNt@>%C()SeR#bKSK=52Y$gIdchWA?i6FfvG1DVcF-vQMKI8Ozv-5ADVnLCE}zx@0zz7|T9aT&jrcnmQAnvveDdN-2jKXSt-bmh))kD~XP9Iec!43M91odeh*ZNOCg|Nriioaz!}^q8*dXJ2l!}h|@cziS{r|5%#LO(P)m0PLfMB`vjgX_jH)gvH@a9r(bR66~@}P5o~nRuPY}3I@svi7EW3hbI^LHv*N2J7A8|}9kaC1XvSbEPo=n7*FpxfHkL_w&^A$DXp8AeHg1wrT28WB5}dUs3A)-QQ(aXjyvn*dhpcN2zj9KMsp6w_ZdBrmF=KoZ2&w6eYC<{MNZFVq%AuBF5X!CFKw%(iy?%x?)cKh#wOof%N%^?xj=J%h0SRB&Q=_0Shsh%n25%Ip$T&jOReEeV9xIv9G|8=^{!njZ9YjoVeWA1JTetx0N$DD8&=10{@??ffWh>HXGp9(Ohp}_QuV@ww)^%+du;nKltB>R5M#Uq|&TN(`$@CU6Th-z9j*DI$nw|Ph-i=M_oMQ#!LKet{*8N1Umc=GLREgQvjh3W4$JB9y1j7^kPMulwU9F&2<(cZ|Nn3C+lPM?BJfY}$GCIcMjMpfK0SUTRPOeKhI@F-XEwaYS5Vv~PzKcLRf||y4U^ltS&K1^NO$(?mtoQfQxZFIP=6XR>jVtQH6G>fVt1G!)Ygn~5h=Qe-a(c7rb!nPpF>j|mnW!<mg=v8Wq$<T5E#!QWOs|yjYS1wBojchXr?kv-wsIe-$H+Nit6wMzXtQU=TWQt2tyIVBm66P2rfU~XtvYVXtW{Q>%t?m(wdzt2^YhkhjYit4>*D2^u~1CHAHSZiF+88~O_OTh6kGTl+KIzRVk$HepVenQ--X&XZ>X6}H+!B@P5bsJU2A5f!I+2aaZ$O`pVJnd=G<ILXKaCPK<kV-%hvUBqc=?C7I~rNFew$>z^Q9i^YO4ZM*3YOS9NL)xv7&fn42Dp)evRE3oPF)MF*-?w@xO_Lrfi=u`})5R?9jJb&0D#$|h7YWJb&l)`$sLO5^Qzj-_N)D%Ds>tYm{_v^C7KIL}vdzWIFVw9^t<%V2HhYTQM=rCk>><M}{0pS!xUSfI%j^R^35d6ix&nj|+~oGT3GykR2s*cFYeGud;teG+a=Ii~hb!zoovhSmOh<+;#yrv(KTmQfN#gLVy@jh4-1mPfNWpHJmA#gsF<uTu4fJ5)n58A}3F7MNp0sZQP;g|dzOYEhb68@YMiup(@wqTZjcwpxRTry;E`^1yHyTr?fEg`<bfEO-pJb9P+qS`SMFSzM`a8#e?6%trq}T?litMu>{8tFr>pRbNbTm98041zeMeq1BOOm20bi85?_J^Wi$ImanItMLxEc6{FRjo$6@O`Ltx0cJUxK>gU3SiLuyK8YRwva-E!&H;hzLPwF~yR9~$o<BY9c9I|+MBag(E`c;mKO}z?ctS}Dbb?lsKs_=fgqLva??>z6Q;**F^GImxCW!9CR8FP7Q4IkMIhKk1-D$S!)OQTA9*RGh#Zrq$lnTB>@lKPFIO?)bAmn0XFX|w=HRItk~yg5pXZHl3enpV_J3rCe!;+@c_X6dPjwL@B{4sdl;%XW~`vhHtp-2T$!yv=yy?ohJ3IG;H&rj%}%!y$B?TX)NDi;uO`y2w&a$QA33*Je+ojEMkOY)(TdHx{LB9aGoc#nNQ@0;|vK8*QM_ZQC+Rrh)+*BcF;fP1Y0+WR22HT<?<xbP}vFY0V9@0Xz3|`PR&+O`scJMjw#|E1pHAq)(!HRF;}{5v6+^meUfSCt6tQF@s*xvajN0?S!1k)GC$QR~jjzs#&<FW%LUImv-k#Cz`oHtC?&WW2#ilI+#ZMh0HWGuCo+b4LDbuQAkHytxM@<x$&vy5s$s!=#(y8fevk)Cs&qHqowz_^WDg}ZO^2Vl%ycBkZx5pRx+4_86Blo>bk*{)Z%Jdad&{Wn!6@fb)lC@;@Kj(Fe+`f@gi6;CuXW}wc1*QwEE#DnCM)n^s$P<bx{uKCO)m*+8jjfodpw?Q?_mcGuDHFnw#wtJ>R^MQ6WyW=B~^V&7KUbxs2Br3QByVa$MSPv*!upG%xcE9CC-Gl}t1TE4?;vnt3Wdhpydm1iJN7#^I9ED{5U>HtbOb*JJUTGpa9*RN8Sl8Z3lu@xa9CHe5)Pfq2F|tGk?ZV^(b?3r0%SijYY0qLogEs^xJmy;RMkE4f12ZLswr+a4$_S;lIY=^Kf39p>o0ptZ^H)Zq*lG8S#QKAVs7Ds{y!cPG=fjNdz6<*J)=Q!O%!;Ngb9xJE-m`EcMQoJlNv5llP7ZIcQpelF|9T^6+w9~0>urCjITwXWF~bC`UMs6FJjn%bd=wQp9_%yL<<MPzlqRNH5}bEd3Nt}p1>sIf#l<yb(URdVsts)i_l4)3?<I_Y<socUQM-@C{v`>1K1?3k9ojZjt-#<Z1^=Pli`r);JghPr80teNb!%`_R8qVfPT4A7IhLpj&FO}2Tp;8t&%THHp{Xf4>iSZ8vQi<z1;Pv%I$&6O(=1URU%d4V)J!c?d{Z_M@CMqF~f8f8LcyolABM8KPou^C0a%$8>7g<@>1TLxpj0B+wTly;fhO9z*;iRIievai!I)vOqrPwX-@WeC;Gb~|czM?L4QT(9U|Ip=!DkTe(rq4*%CFHUOeZ0Fo-7=$fBQtGj)16WUQ?Ion0SlDmX*S&N+s<oC4qtXnwCOZQsW+za)-NP87hK5z?%W;IFAJt-{ML~9&Ir?H{1>VP8MKrN=RuRUfo%S%giuuYur*A2#cv?0Zi2IVaog1c{wQxG&RnJ>a>mpdhW`mq!tl;cvjx&y}hSIVS!pJ6rVhlm~#=3|->-1^87bBS~wPEqBw{4J<3MuZq@VG4=1+Fg0Hq%L$vqeLy*hVr#$~co!0n6GfaZ8nhRU!_w<}j@=$7VpIxbu-sYE)JG43TLuEnm@IUm(*SO4|n6EamFh>2yC3wNDgj%9dj-xm<oZw1q?27y`E<@%7YStE&QwMqR%)25LIxc^gS5dfsze7^$Q;T(Om5NlTy?T97hUpBhp|X}{DkwuuB9y$Gz_OorCEr6o?GXKgI*ALqKc3f5~zkl|Ri^d&<wE#gKk83(HNHJWj+2Q8psYy3jZ&MFthmPwAc)Zw(v<;V;|ip{LiC`^ouzB-|Rk521nX00+cVp^@NHJ3;S*uh1%d||#Ajor3Yqd$)7^7>@9=p-cVfTBm))Dk+snNNd3J<@0u*8YIB=tqK*^emoe4{K2}*_&U~jkJ3`la}HmOP+`((tSc+RGHeHP%Ec3<!aqo8r2q6^U@|#&9t=xRZLebvu$*upWCB7EIbG{a(KzuciT$Mc+HWBSq!n3bUlerxtiO6qKrP1E_H%AwQr!9Nb`+sNjH&aI=O0=EOlh&krbWJ42qlsGx`*ch@&{Njr(yt8kiP5vZ6N&jM03^mky;jMoG|?9`sdyYtIfm{b9FYmyg0TU<^%7<pIo-ki1BHUAfVCrk2a3>2QcDN^mVZ2;&ho2h94JDKc2mc)!9a2S|1v@=a+rW$3opB4vxX9kWhORm+V0n{}?U#>b&v8cT=VnYv6;$0@m^&giY4UNE_66iTzO$3$|J)K-y1CUoe$LKkQ@(62PB=Vq_8s`pztBP|^nE{2naJex*KW@ml4a^;$HCY}Mzl9u_#0pOt2gSaYZQ1xOS8=J0HY7*8H7+s`uMb6%JDkP5X$P|;S6U|Msqik6wCaXm135-4Q#)7p+6_VI0w8;BavuwVT4}}#r_bMJLtg?Nz&&G6h!9Z3O4gi<eG$&9jxP+NV6-2BQE?a!ct|jiV`RpO{(vqPpv)rJ&k<JE|VXisR!(RoF=35mfBk@ZoPI_Q5xYfo*Rv}jyZBih`FJ|7p|H4fY(TGE?tClI5#a%KLC<B}DD6oEt3v-?{mYucP_Jz7FFUaFeU=&LBCC(K(TH0nxRG-=C?76IUfLg82emJOO9Z1ZU8?gn2RjME8R-CM7EH7pk8!n2Q3PGv4FlkqiWsQg!=~`M|Ni8;k;=<>O(w1~GU|Q6gOo_IsV!1j+CDxe)<!cnj=7Ome7}zCl$}sUNGe&3bqNWZMeX)wPgOnTTeo31!s6r%3Rn>B0tgIFKc*Ga;O~#FK6iu5urb&qcqnW3fmc%62#|AD(DBM|)jfo}G%1foBf-xkL@u;VhD)~ztH<%<<IJIs|Bl=XcuA8o^+;U{W1C^R8IT#bG(9l7<RbI=aV39=ng-$|IDKRr9?yBm#o~fzib*cLyQq~LF8Itq4YOZ|UmT#4ExlK+5f91kfA5E%6v6mjJC6bQ+VkR*IBhkO~aV6<S7o0L7sjHX_8S_Ig)9%@;vOp5GX2xFNTI9Ti%xc~y^lH5vGfqdVe72S`Goka5L|zI_*d*2StLsCTnViL^dP%{`OxkL%(jDojsDQbw4&j!p-#5hCBV-n40*i5CILyZeYNcclAC{8RLB*@eDk|RdW@9RA)f`rX_FTK5{Yr^1J}70v@~Ok9lTKQS^K#d@Kzoxi&E?9=dW`UAJcN4SFySUF8|fL_fu&2|UsJ(W)S=!iCbeWR%S=65HmG#~lYXLfD4hLdAyL*RP07iHBkjk4ZnolawWXt+6@v8Ct<nxXM5RHNXj3ZRvz#Zws#d(rIuyx_w<vLx44kE9)?wjfWLQ)6Bl>f@E@a4-Nn=E9oD^8e%AaNw6?7t_+tI9r(w4h2wh)gdB;K`Y-ED;&XhxODwOyHjtT1zWY@_yi6`|_UQlu9NWG<L&IgWeO8y}~V%t;)@J(Ad9X+Y=URottR%Q-tHv4qF@Ze<XbB5tXAX)=3FvXK$j`RB3Caxfmw`kPEUMb<4+M`Nl;)*7phOqFYz7CJI8m@?iOTTPhFzF6DIMh2rK>I$nvZ7G#=%NAWndtf}5XY$PuQ81O<NN+gY49Br0(_1#P?Lf1LOWoGGeo!BeMh1D++qA}8@v&+pN!p_q1H?J5nG2*g77ZG!zPzs9PXIf@JIqkrqG-#`Hd9*;$TM0lK?|g;j||x);xbJ+h1<ob{ccNal6Fa2-7Z#<1)bDnZf=|AWG@|vCvsuWpy3`2^Gh9T$tYr_3v%Y^`zz-czJP6Vo(|Ly_tHKtv_tGLlgbw7j#V-{iulQZFI%pv7WoZoaC4n*)uxZAJ(9p?)WU}7h?)Dm)$<-cuA0$|Y}qq49SEL6sNN`Hhy-fUNXs4$S)>+9sVqtqx?#F)@APyAU!PNy7^AD$ZB=r8n?kW^b>f?5zEHKN-3AtSSG(2N`6Qr@hdT9sJ;$U~>6|3p%#w~|-qGoIQ7%1OHuGVvu^gVPX4z=3Z|S34G{_`P#&R!f#(G}8ENYv(-R=CC9;apzzcq&&3dlyfA=i~!WTmJkCWB(u5OxK!7G~&Pc5FtQ+_*@!CjO$bUrdL4l_Y&$&K6rG8tJh^ByOaO><H~9Qp15dJvg8GrFb#WE_h0zZnTS3NX8|}a2)9>SCMmt%$39wxe8P41YLtkljzTM)9^5_EGN6;6<TVayGQEvy5eWr=epD9^66sEAH*`=Y9j3oVEKsGV78o~C#O1t(T1A|qrGLZwRIMQxlOcMcB=(%wyjNrmC&0FdW*qs)LZR@UT-wC%^lo85ayP?W7C@m;`F&vFqe;}QxKTgOWa^IVdf-e%3~H?5;N!w7QIey$4dIs=aRufBpt|u+RD?SH=OiiVxz>V=0q$>2e44u@8wIpek|%w!f$yj7zzdb1uR$!rV4zCcUJ_fAkP(Y-e?NT2ZI89i%zdOvFHr&j}iVcA<yNCiJ;KA*ljB4%?0xUIH4q43>Jf!w+zbfl%`XODqmOtCT_}Gh?e*|*+L<@_vrU#y#8pRiUGMCO#{UI-c%qO@D}*4*M<itN}c9Jk7!Pe;#cquJeh8uzyosrTq;<`{ONR#AJ2MMStnN2g9ja~g@tn7BoEv2TqK=~*3v0{jJh+;i5dLU!#_s&WYL`=*SwHPgGz0(HBr(SLC-kUtT~}EZbFPW)TFapG$&1b(&}Lxxgt=r-T>bcz&!17DCqjRlRN)W1kTbV-NjgZ0$LDSljfw04~IB<X1Bwy1V-V*W*a_(zFHJH0#)fQfu0%6#{d%a1Zy!w2R(yRIqQfIpzRP}U|kk(V>H7KK?xX@KNrmuz=$&Vq^HEN4oMLhhMchk$51mui9EOCv^c1gVrb&JAQKJjik^F}O>|CZX0AjY@CdQQWXjSf6LLtLaTB%!zbH@TRjquemA4D>$OC!ufjlo?foT>Rx5+NSO_U>m#vzOz&h;p=dB*i{u=tcbo_sBUBN$0@IUzVDf}Hh;g$QE@vIQthD`z-0B}9;?r>C)eI(7B=r`zBDdHcfW+n0VyPbupEe*Ws?PhPwF;rrn4<*&ZG{q9RCJOvB?@k>y${rIKr%fEsRe*Rt9=G{MDzWk={|Ge<;02&XK?3|DsNplmNZV^I%8lFSMUiSvG2E+iC_JbX6$VrD`PX=I?PDuKMZ+b>fh#?r!`Tek!lj(>8pA^YWkh`5cJOF+J`O*ClI~0RYkQ1D4gISyaKAH5+NUlTD0GR^{?sdb@ya*9Y5<US#6TzLrF`V9QvNt3AFu}a-Pc{ESoIDPn5m=vNXnwLaC;Ldik{!$PtZ-d=Ho{rJ{%6BMn`D*31BVt^h!n8^<ieN%|0py8D2x~uXO{pP-+WF^dKirZX5}B@w-fcr>DgpBI=$88Y|erqU=w07Q9?Y=+SB2PQ|>bz=#%EAEP>%%izLwyPH_Mal5PX6Py_$E#k9fFM^C4dj@Bl2!Ek`JM-&dG=ejq|i7-%sI6UX?5eJmvQ)+_3DS0N1L_+iUf+=rv!Z>+&{7CL#guv+?3H-oI2PJYsfN|{l+@G!%Hxm(JnK`#x#^5VJL}W@<U`iG;C4BSfe|JI$#1eJ}7lsn9z_Sj^3}K;iaM-V~?)WF<U*-i^V~-)25kV_Y=RlXy#GE2&0zEywGsjwv?4E*^1~9l;9y}|tpkqM6vjD)m%oiv-U}cg7)rNg@C>#fJtBt^gJurWu`prvztjv5j+6M<(V!tJLt6UEroa}M%>=@f3x8(+qLXgU$pc8pwb|B6G>}z56O*wmby?S>1BM2eLLA;PsE(8u*T#Gdhq6Qv0UUb1Fz{{~i8MbSSJlbCeU@#Lf2(rbU@f{15`LT-K^26HGaa{ZV7wou6o0#pRM-eyJx=!>0p1^cDhWd`S95ZEG1jf<$i0d(vYh8`c0sbqfV}TXR3`aosbIK;p5f;-~(*{jjrcEdhZQ(PXt$|MGc?5_FWKR|B>uUi0KsTRA(37KP^`5f8Nq~87N<*t5$g&KpJl($Z;`XnvKK|mJCojBs`RSWizxvkpt3O`7`1{LWz5DbJuL9-#<bU2fy-&sg9L~g1T#L0GzX6+zC2<NEYnVVECCf0A8+&2<#vitC{^0VRKRx;4!zW+<=k~jwUcLR{_KP0@)8Xp-f4chP_qX5s2)q?ch$p{$>FVvbp1%7JSm*D5xqRn)I1o_H{?!NHyL{`rk3aj{p4G6M20;5e6C(&Dob3{momk|tx-rX6o<BLo@ad$-up-{4plAo-;VGEMAvrk}_z7|(6d{@uDB~e(-c*DlojZNBTd4agn*!`NBX$Tt05<}L>|9rXiUDr|yrvU~80hQt?Cg{m0J|DMZclCrAz_kBaEXXhEKcjIFT(%d{3EoyKK$PH#rL25{iE%d-#NYMThtaJeV)JWW*<56C0ug=5mCvBv=ep8H$+I7bODJce|zQXN3ZUoLM%o^YeKZ)BwmwL;QZ4^4<X4O?PGOoZNS;MUPwE}yuDa<U3@ca1vZ1L<->o#-XIlU{m0W6U;DqG|Lhc~uRF8IlZMx%2eU7}-M7l`N9Mjo`}Ai&y!zc|A{l@E$L(+5diw46AAj}X<q!V>*30Gh|KsYjpFjQiPoIAF6Jg<qI6@nl?{SaLK$ac$`7am;2TcX*P4FI$&I8{>dLoD%fcN(0mmp)l{PXSazrTI;Es@Y~z6MzJ>c2j{{MG0G_w%=buz2$Bd*JcvXWs#pfiLmo!;b~g0LJ|Iv(FxX@gqQv?RUSp{L#li796Zj$f&S;lHxe%53>~XFX6WlLy@iJzCAs}*#U1bTrUD10u%}ul>QiYHd>zU&$D3G!&djqw)k?NF3r<Jkt%d1N>025)*O(Y`(jgbGPsePm*4xZt3Un}%FlP+2NJSHP}GyJ-oASO`+z$ELtcIHA)wghkADv;{ht@Ub@|R~+gIM*{>Q6Vul()ug>OCi;!jt<|LpNsKfiqSTf!7<Km5hj>mTn4&@RO^aq8sSKh>P<d8Yre1idYaZ;4Vs$$kG9ED2!KfKEnad!L|poaS3?<NF4|gZ+8BsR3XQjxG9Iit$arT`$ZTIC*ffzwy!b<?mj;_Up@cKHa|aDPS|eeoue$m&*@+4Mgeo&o5m4`7?p|k3ad#<@Y|`lVyjK%&Vv+cP5zrG)J%#7)F6>Q>Y=X5NKeZ2wLuJh>zgE6B3$slanMq60AO!?2<H2Vc;Floq#Cd-Pw*+`AUK^7Tzm4f&!+m3iQK)57l4}zwMFt4#ay}9N5D%vJIEmO;qXO;L$_FgGYxnhJemyV6-G01IF864E+!^ezbEeZsJRP@l7ws+hX)|2%7?n?RbTqsEHe%eDw=Jn9E;$=jjW-5`=^hg1CJ9D=?*Cxx|gn$4ziaz4EK=i|<{&^)bJA{(Z+@17HgpZpO1uWPrhrSa;^_8l!#OfS}(^WGR>{L^J7J0rN+^xo{9h=z`CEbGjaF{}8egXp2C)9<On#-x)A_$Z{=$##xftA(t1QpF67g`IQgY7pg6|PI(V@*91YHWfUA;Z6AFO=>GWAmjoLRO0U;H+Wz|6V39p}ffw^X`Dou%72b-AY+oo04oZYj+MXmiDs`nN+Yf&C_#eO8e)^xHRQ=#L+t1$+);{c&M-HHJ_441IeEjY0r@!0&6o#}OfBv&;wdF4cypUnwa^syjio&h~kSz-5IO19}hys=#0lSf5TfnXFE{B?vOeU>4$@6kDMBo5Lg4;cph*pYWhrBtxD+AT{p$UaS8aR;PPY7rc2Ks>h9?=?%7?KuUU(HEiIvg$mK)z~z+8nV=YxlU{k1+27FMyRcT0-BN9_<(N*xxV`|9IVm-meJ;NaGv{7Ayb)9&Q=|QZ*@{7=c>^U3<>BYqA^P@mgn4djLnI1qj{5TLW&_%m=FY#lmmUp2r!Z@Fl?>fY{AIa4}(dT7Cmq;2#NUUW;kr@f$Rp5iMwv>>B=4GxTi8yw$UV%MlY8;BGTRj3bD)pvvL+&)^#{$JnF?BUYSJFM3Ms@@Cl{)&vYiu*~Ro3u4hCMw5fZ{b0|(;JD}bCw@WPu&fer#2NV9(?rnD`Si^>i;qTN4gU5mXaRoo*^{rn022DiyYFB9_S5YLKfe6%ufVW*^3f}oZ+#$GgMtyfeebR9D}M&Y&-Rt~fxG};u0H$E%b)-9$?w0jefg(*rp``5?u5Ch?nIMIa{_D_a>dYKodPBxZW(s>8EOZWzOjToXdoD?XH$HL0c%yH!me4-o6IH3LfKsvd)*0o4PObjC*XXb*W&?Afg2B$5k9B+=p)J-Ap9s{n2Kj7is^O<R*X{}BKeKgK0n*|2yc-Sl7Jz^10-+sBDbd^iP?mqFtOG08Q>07-CpsvemkIy5RciXp7H=0A?~j{7!V)^+-L#}-(mS!;cj;exeFig3;)U2gsMn$Vgg>5O*Ob%7p_#SO%8Vi;oGQo!0*=YFhI~jyM~?Yn(RQ_tvsF*?DBYJfRE{J17ZfZwhkKX!%&|LRF=D~m?pPBgnb~e1+>d{+kc440?PDvVZb#c@L`MC0fJGb{qmzd6YLB>XbMFL|D8MYF$<-Z9s%nbtmWM&*TY8ys@%t117^29S%P9vi4PST4ajo`f`NeIUz7@vhPOn(9fm%P2jhSp=`O5uCVt1-M2qAA-rTMA1g4j-CoIhC&xc~Whq_0Ppb@qMc@sX}2g9^?rZf-mOfcuM5yO$ZJq+ex?`@ml04#xuH&tMqlgEbd;-T*Bk?N3RAZL)!G?_31g66j%9%;b8Zy+FqWpYG9%=ob%6$TLDn{|GtB{9K4TOrO1oAt;R1y}$M4Tc~wFnu$E;-%klqG(QLd%a!L;Wi<3XLq^=gqRZ7okMn?iPMG&8;EyWkTg)s0!KUqGMV2I7g~?bT2mI<$vmbk4EE{H5FzG0;scAfIpa)-f-8|D8`TG)CP+JA=knW#z~Vd=$A2?svJGP)a7PM7Pj-}kCR7+`-v`M}o4n6O06=_T96~ysFgKl?{&Vm|SO6&=5<#=f6qMYZe(4EN`cHrToy)&||MHzbT>kvMC-1)Z_>0$`{^l3JPX{IeFdMEue+z~`zxdDX_g>rn{V!Lay|Mk^C&0*up`;gIy!zo^pWlA@;`Rq$2yJ#w-@z)Mjz-jyXY?cCIewSo2z$V(!%4Qq;vFG0ywBgfisLX~7~J@XfC9=L`Qd^>9vmeaPaM$Y4)z&ez;B}+tuJCI@Yzix;JBT&de8;p&qrW`c@G1ug#rFNKKVQ=@vy**@ZMxJ6+V4%@5HhM@P9ah_mB5k{SbvFm~^OVcG@1|p}nrMdkYE?<ar1O;;K2(6I$DGEP?&I!;=sDMFR=`5KP${EkJycZ-VfV**qRBJupicVDr+`<C7q;&>6Hk=(bWRc6FR2J5nG=B#0<{&wyTnfF6QH^SlMpmKR3P8+GtI6h%8xg9v5MTz)*a@`xN0g+TQnLLbklqu<qm^Ygp`Xbt4(g6P3n*;xh0X$d_FuE$VNmqF>SKIt8;*CvKgZlIzZRs$Ljm6H=Ex(5Mv$uGWZavYx>;Oz0L96+FjM!<037p<`LVff<!@^LtVM!cS!o<Nf8M#7_jw>VXN&yZeUKSTV9raKcWgui$!k>~Yy=ni_GQqZUa77k$07MLBN@d@Qa@Qiaf%i>FZNAidv1VcE`W>RY5(XhSxOj|9op;{QQTHK>;G7O~@EffTKq`3p~7IOSv7sT(3?+6BFZ`o&155j;G?46>N(3ab6?Fuas<h~t~TU$6!e*f=KN4zXLlG|VZV*9!1LIOo__1XVyKmFbFm+yS<=}+IheBt-o&;Nb<$tS0`X~~C;gpLo*L{LUt$4cR>5^es>h~|VD!S%uIpCGx>UE9}xDTK3je(qOaT>a=h*c#yU<Xzy<e){;UpI`mxr-E|}l-854e){yguL3W3``gzDdUnQld|PJ$^1;(E9cJI6-k}IEgMV%?6F-lRRWR&%8yEmZ1gM8YhX?e;kqbuqtx@vGu*4i^+i;td9YT{6_CH~#JU_PY;>yD$X&@K{h~EsiZ_vph>Me@l2p)Ij!BKbdOsIkE2Q6-bYY+Z~5CS~b!4hI@PM8N4rw7OS7f(3|CK<oSq8Vs*(DuA_`1Kknc!WS9%v;#OK$(c$t=@Q4Z$Jxp-k^C6a5qB&Vh13uYwz}iZV-B2c`5`_u~Ria5;$Dsc<W;uj?dU{WH?~Kzy>$H-6G>byo+akHQ)bphsQdh`*9PDI!2G994H3(lmNitEIfZIh62x8+>AdHaXaX>AI4MeKOBa@wx9j&>a!1k_yQdMyujbvFaG24CogPYeG6{#z4OQI$6xMTO1LZf(Mv$SJ^u3TC%~kA<-e}p{=wDHe{=Q0+n2xk`;n*@z3UVM;6=WLOX-k3`03`ABaE0JXNRp|{kzp4o}B^Jq}4wXG|#o>0F|#_9iymtQ1LdkcEMQX(a7wbs9_>sNM613%k781*uMGE)eql?8^~|J{rHnV2vYy@t$$qp_m3`r^7_@gA3px<Jz=AdclEZP{%HH@?=L_4;gc`F{q<48J5dN=A&Iy_?S<+MHXs2|gXUyNOnOWky?w&`_No^NeBI664))<A_l6C3C*yVj!-8-mjDmDIjA!49U-2ITvV8nX_<AH7B}TYR>aPoK?IK#Ln|A+Hyh{B~N2}DgV^w^oUl%44Kg1x>4r|Q~*2rCJ<}PO9**h4?IZAWo9xj^OymAEI6cAw-8WN`k*hsh8=y*HsrVZ~1-M+7I=eojCD0u4@nB4`4XEui}*a7}`jkKGnCKM6$S)KrM2plLJSb6kWf}{zYRhl$_%o^~VBN9a=|4n>jG#tJ{p3wo;netxUeZA-{It`@%ebDwt!`(9yfhTGHP{e)xSTq*X@zGdJ@YOwI5P19l324CXEYZf8>9b&i^<+8(e?NKf);;vWTi7mcabG_-Ywgx-?2QJB)Mucw-yKN<H}%=`4uCzgr`vW4Z*0PAEhZH-PH&h3QQt1YTX%@Wx4wrwz@=+~exHr3nul<QRS#4Y82IX5O}=k9?<NE)nTHQF2SQhSGZhaGUOc+p+jK!vBo5dUm~5anu)MjW7N*cKAA3IN{cNutf~XHFh!c2N<rcSzLFz|`XaNTvT1t1-G<VlGfyV>(DhR^;R#26KUh}oC*$H`(a<?e7=QV4NsgZwLP#$)EBhZw>ek^cqln;eEnwuKOX7#}>#rg<%=)ko<!s2{PUx4}OpOJ*Tq&if|Ky2;fvZtpX+}7m>q!SGi%aH}nH{lL<oo;or2lWB4G>-s2*ql1r&72^+J-#1t<D2PBU;t7n^snNF0_R`DJNgeEA!jgr1StBv?Mwr8x>kQgI`I-0-i8ywh~4mj1W+QNg8+<(Hju<e=)q6fF0^<{t}BJF|D?!10+JUjP^8@Ni;F^Rk1+)}JeTdx)(b_5OFUUGaM@jpKa)u-!M{V(9uz$O<TH3Y=vUtrPJRN*9#|jn(9XAh_Vk69wy(UuedX=#2fq{CA0cY;^w&S<&ky`ycPiknT@M_^=?EJ#ZGt))4%ivy`}K|(Oc;Mzr^hgZ{nLpzryqa-xaM2o=;_(D5lHN8#IvJ9^+<1^9tLAKL*R<B*KEp{oO#>$P$_h%p;(l!Pq7%+>JjZJMJV-%241Vw6I!!)4~fTZ?^4DAeTYSinzjK)@dFepKSR!6)9YwEME&sehEl+iJQ&F7Em@9rffFwr()pUsd4<AhW;jd2(+k|5%X$xxJBwns2gNInTX<=Njjf}{OAk@tLpJaW80hV>9i!GQybs3@xPqjg#gsyY<Lmp7MV^UzbK1F%z%djBL<a_nI=4=k2w}^-&$xZ(eXtrY-~OM=w?4l7@!z&@eDvhoUu^&Wm+i+d3C*^D{?Yay{|#dluYM1n-Fx$QU4kZ99Lxd+PYf-eeD#y<>%ZK7^6l-PUpThu2F3s&o!x46e9J)4<FNXHF5k&depJGxVvYEnzg7$d6Qm!_jbNBhuhD4?#{>I99CFu*`1HR7|NdPi7&<-dd4G7CJyecyB0Jf&B1R+!WR{>1xg!XWT@r8})JnV(1^VUz!2OZy3`hsXiBf3yP!BZJBPYDve0N{FO}2Q(=iOs`LNherI7Ybxy}y@mw>ibY(~;<r-Bq|1(`A_{zXi_U%~qQ3*})T!;tN|OJUY+V>T3dK8r-uLMDZ6fprbFu4$oRW27uc^)$sSvhhWXYJLvn}AsX=x-mw!?*SmzmHa{v7`oJ~;Wx`I#L+}E5xI^X9Q9$%U0HS~Jufw45Cg46)WTHFx?D+f$JdN(r8P36o2)`*qPmWn8@V^Br=p~%PfTrgdRw+_I(<WwsY=UkQz8{CMyaNFY{TsL5DOmK`TK714m@YZt&|U34r)1yw-haJPE)dOO!Sa}yz0djRnO-G;15RCMO7ld=sk0UfUIL*qVVTiVDd3{HO$uQKk5S_ua_knAr~~;u&O53?@VPL)B6t5P+6VOTlmhzguB9OuM!Ul;BbMpHTaZAQ4}scx#)j-2l%PD}b1W$DC*>JD^vk<`_wCP$Waa~<XMDzo>$DA+@@C?WnqZn^qVFDT`5mnL5hbpTYw_?9M7|BIk5&(;^wwyqMD7c12!}GkGHo$noY25{USvuPLOIp&7jA`45b+$*H-<FBDYA|6DvRIO2jC7Q+)*oI&+L7+b#z!+FlzQq|J!HPmHTJ8k!NkU-RxEYw~AeUU10r)HNC&zC3ukR61}yRB>+D}gP}3r0)uD!cs+UVej*Ko4XP9t1n)we@h7gyA;B99&p&_DF{0KZs+-3<Jznp*-o02TbhQuIg<f}mWKINgm}K|x(i{@x`P0_*ya0Z<+rd5T1v^xuM@F${`@%iF=fqn&+qXa6e)zNPi+|dF4lh$}-+T@FZ(n}(_|w;c!TI>BpTlEY|MBXRzrQcsc6;*07s9O!c<lSjpFjEQTid_Bx_$4*!u=k2y6ev`!0Ux?zPA0~$NP5zTNE$$ls7jf_AX2uxo>gnO2)N=s=3Xtd$R%h0PHrN;uJiWg<+^rAUxcv1&BU{$zjmz_j{=y__-0LWsgMx*~QMT3vV;)bN|ZDDgZo0JEssnwuXNn=<Mxt!CU5$5YCF<@P-n$P@kTy$X)IRUX<QbYc!z5S^}`iaCEE|kc(hB-G=$>%C|tDLE`bJ>8^W1_FqGX5Z=1D*~|piI}j1q&30WPcM>dI1JvA5<W8ksbImn5L;_!DU%5LJSbU0T_!LNx_9*-#bb{V=#UJ$30NnXhL)YMiV*ZdnpDsv4W7AO^dX_@lGwizHFgMK6?rsRjqki4pHD&nXo4fu40)xk5hN5J@;lgjkqyPlpjPDWIzb`_v=)J4p`BS0GQ_T@+x_5W*2*pox!F(|p#4_n<su0Wxl-IyczNTFwS_+D_dAf~yKtrLYyaqVc2)U|`V6#6jBeIR|T}a#0sXzq2@rTP#Uc7qqrOS8z#7=2pi+cO=+uP6J#pfqqe*5yrzZV8A+>!?FwQxS@@|_o+eD#CJpZxanPw#F&eD(52|0B5O`;N764;dK33=0f+rSdKzku!LgOerjXmB^2WcRj`X(Lj8P;DFO$cyv1o1Pu!GCYO#-_3jOl>&w|fi-)H?=}sSMcF!W)ffDf&HGj+gt`>uLX;HBo@xI~neAT^lTSxrGy{qmhdeD0BHZwe9_>8;!2Y276G)2;b!&`=svv&fW4<8DrVU!b`8)*2WCf|6q+#1b4E8OopE_45N#pmw5<^-{Scse+Jw0Bt<*ppLsx;u0rSeMU<ew1Kn4w-A@VHuR$ZIXjgFXisppNkfvesAJNH1Idg*y#u|Ek}02!qNy^DYtKw#))N<0mgrH*Fbp+){#K&%OC!2`|`)zZ@qK*!LJ{G_3rkKzl+)eLIlp1@Szl+^3gC53rWph*SBP*NI-F7`SpKyVp|N|A-i{G^&U33xl0c$E`p-C)8_>$lkEdC-+R|jAoGs?Lg&C@`QNxD$Xm83pLLGENg>ki*7+&{UHEl}u;`ZxxI8=+9&c{^^M|GcclAsNGQrGG%G1`gjqjh15o0@|(+ytosZnM)N%6dceEn_yCm;R&^3VScFX=x1;w9lOfN+IaxZ2Mj65=-t_^cViO@Du;xSt$_g&Vx55X6I1%^oVyln@3XB(>5ZnhrJPdV4EROlY_xgYdY!g3TMAB#eLEn4z~8nP%5gU^MY$+F?1Q>z;|YKNtMzJuyERJjZt(OA`<gZ`I8BmgvEKQ{^>w8h(DT-AN|?;@~O2cd)}jyu^C?2u0zA5D4T{*iqQK&w2w;Ohv}<Yiv(he2tbq=u*^Mk8WLu0+^uJ4+nQxq0kq4ymiJ6w76^e^I49RVuO8cB#;WZ4dQI)3G!2a=PhqI35WRZRuJs2Yv1Bl*E_`s=!*hSyLgZAp#c-y;teoja^#dG81oU6M1o!ehtH>vzFFp(%WwU9``gzp-+y=ei|=24@E%O0@X71fM-rdD_}ccPe?0l{<Lwtef?nAtFFg6~|C~N6+stvoD+K}z#xv6tRoZR(Y54dJe681=2@m31_<Wt|^N<=s0t<9Kc(BW<#-~)fNkY0y*d(NV5VIX9cWYL2dXQE0RLmxM&nzlWlhi09B$<Isqd81TDr6r$*#*TOz!asb-P7F*0YdN+6Gy<?pLYY9g7QXh`K-?jjqWwxhdn@V<d3-Qg#zILi1_1RX&?`AARPrLBNBaRFk3X>A6Pe?n+N1bbN+Ptf#f=5^clf68DMbhHFq<SNKc;r;ngP}eM>m_`Q*cQuRi?o_U$i)yI$}V-a#q}U}*p5h09;QC!}oIe(>X~KmG3UCx6+umqn$6(@W)Ee!ijRI*15?3Q~2*8^rt~6C=!mPY+M5e7$ah`)8lNarMOyu0DI`@|`~j1GxO}_qQ*-dY#Sb@|9OY6ub;YpCl6*puqSufO|_z%&&7D?mocg8S(C;ZcZ{C2qv(kHx&&9^M!qf2Tl$@Lx2pxXxxjBqVUkd2nrdvIev3-@c@y7vpk3M4?HQ1;wD33X^X=VQUsyTNFKC{wu^kv??~LD1#%sDIqXraCd!Oc;U>oJ&O9H%-M;?I%U}KdR73M=(69sTi{7V`5Bg|POltsO1M&FsOHcpfm*5L@(O&=M?oK~1d^P;Mh=HKp(uU`F&|5>fIVhyzj=qB%hDTrEE#&CvJ0zGMa<W9lJZINdk#NUE$h~#>`g>1b_}S4Tt=l(V*}nL<?Ta65Klwe(==S1!r<yxAkcLDX+uMW$z67xVt;NKxV&niUHsDI{E12gWfBqAoRklBU|M~q>@9@^lXJ3J3B7%JS-PeGt%d0=}p3HAPzWV*gPri8n$yfh)`of!!zj*EH?N7FEzPj(sicumUs|2%Id2KH%Zyn{|DfLrC19fGanNENjfCu|Er@}4_?{xrqcpLgntQ&nsh-5EMl(;8y&02s)Q~ugNP}rxBv`5tVE%UT<`jF2${^(@Qi_w20+yD0H{6sK#(e)7k4>ac|!igGmEq42F``!NABT(b?jzxQYeM>~yrF(E!5uWktPAN!=>mJeTMjC$3umgNVp4g>zLGSDd+?cpKD&2&qF+R`?H4~UrAw|gyETX-0SRI(%>RB<RgIM=JA6OD{z6Vo*<3jWP2$6so|5r1I-Z)2q@wK^qr%TJ*;J268cnU$Z#Z2iprtv{JkV|_pD7Xfv94cTN4D8kg-#c|^F3(`Ls{3|zpjErOBK1rgkG~4DyGzBp%sX)av;j-DIQ|YzivfC(gIod2AnrKu3vLqM35i~=KKmum?!4dj+SA{?eD&E+1!({!!-ub4zWnC)jo)m){Lc3E&wyY6-rlt=g947vfhSz^zRnkl!i#hV2^_>L_UN5U_t&XuxgLLy4(1%#Jz``Vs2ZN>z}&&zGo{!{AdYZtYIn+k*N;Mi1Ti4DgR>t1<=dmXTo9oCqdj{^P*^)_`*o<S{`&nVAN^?i$tS>{z53aAfMN0U$NvKr&v)P0zVW;5%Rh!Rdhz|oU;O9gk6#3DK{a@C=c_L+KmF*s{^iU6y>AAI(Y%N3O`hi+Id;hvgerVY6WDug&BI5$^~~}Grv%;R`TX?Y@YJK*c9sC-kaO-_sRpo6VaHSq5o-019^Uk@cfXXUwxCJLHVUy4kOh<7)`~e7_(7ZrIdEWmWM_8Vsk+~r;0TNT9WF+=A%2aa2p(4Cm7YMK!>kAz9WaC7joD|nfi<uxJ{0O7#!)GFe)QSRJJgbId5b#Nd(c8sb%8CX6ovBlcz3+Z2S9gOk$L_^QTTZ0y_mlF@a=PZM`_WAG=GI(uYsnbu-L^E7yMmop+sN|@z?{Zg;f_SKC3g3=?7_`cPDgV9%z^D?xu!#bPO!ieT+2M%lBYV8knG&BH*$;d?BK|4+6k|WP*QgCc|4Op#0YR!x?WA&Y;77>xTk3VPU|AHzA8X5am4(aL)f}0EGf3jsqV5rK#mNK}mDliFg*6gB9{@Xx{MsV1$+OHSNpyx4(}%d#ywFw-y%7v)k`!p`EmY%L=B*xM8eYs_H1OJa>4A5EIxO6_wpC(GfhR&7mSoYluz<st{mc03(IZ7=jLkJ&2J~TpEa`4f6ot^aO>yiY@oM6O&5tMiJHm=<SxmF8zr30mL3ToKnI)LjZCLHw`&nirhdOk{kyE7chq#L3j5ylAspcjaGZJo^D$58h&Tq3r@yoFJJxpKOTSb>64fLaP?oGZeRZR$(LX5Ec@TUY|HQbVf)#qzzcv{>775qJa6xO{PgF)hN-~ceGho{{N>j_i29z_Xn<(}K=`X8dk)ARwxW32{m=v4aTjRMn;!h`DNA>6j13_o1mnb?zwqSaZ^Iuk5z^(pE~b$KC<wo?vwib*P)EE^`pKuj(SW1;-hW-a`4T*Z@Xq^JpMG`q`_C?a``WXujuw!lq_AasI}sVIJ9u=R3LfgqBPpfs{49u0ux5Y)JW~6i?m7++M~|Yne}m`-9KU-Yy72He65x7lKM4qYvD-dK3GYk0*FyN~M)Lq<j#p9Mj9+8m58K0|hS;6{lpNecy^~(|ru7;8A&>>1KLDoEL5`1d|96#+kCe|Vg5h(}|GlHl!a819<yiy1x7Iyl{(m!8{@0?84;2UiASf|m#79P$2IBL+0X_?-`x<}!xOdhHM==v9jlOB3z?-q(`q`^Hivz57@n|o{|Kf&H*c0KG8nnuGMhlODkbw-@(-=1+c!cbh5(lir8xvo&-tXpgySC%^vZURByFqk@{&P9e@D>=0K&)q_K@+XS+j-EAra`--P0)WXjn%_P$en<y2<g@lIi7Tfry@^Dx^sH#(BU1IK%Bz;)8mk={K10bUERc4#Kcs@Gg46z2RW#C1y4LPd6wpGZYt46KcF1U$I*gH7v9F({r1a|8PGh71x^j8kB-gS^CF5c?<e>|v{jZ{DXsXGj^52#cYEh(Cr9>vVD9cl5%0i=w7e#+R^UcNEo7z=EF<DphO6hk{vrg<vLw*mLVWo!m)-GHb}*A2itae@R&KfdRCWA?9x%9Xo~KUyg&BUJ&&=EBM1|3Rll*<iksCHJiN_aTLHC@G+&%rtFM;z8^Vz=h{?iwJc=^4ro_zG__KjCyUZZ#4yL^ZLS%iQ5H~bwnKBL|C2k&3PfC&H7DIa~ded8@LIUc_Ye(bhQYYF59dS+8SV=eq^%>VTaqU1S5y_;x6#Upc^V^n+67|Om|!a<M)J1HV`pca;&8X~b5y*rRM@H&Us`0#Wbe6g<=Gy`I}+qWYhC<Q1F$wwLzx%myT5rr>-$FVyr$ZuR~u6OjsxTTO+r%ediiuj)+61F~|X(}X-8nH}scbVkD|H?6rBMEWDyv_6sy_Esy?&$*pOwav?55z-8O-_TM*Qu#)5>5%PMdt@7Ox6F_-n(`;b)<>H-~B7p>eUB{2w5`5llZiHy$)6qFO2QB4Lw<o*IvSsLC3P?kz}x&*K3&&z=VrQnp?OeKtjk3L(<(~Y%U-ApS)vT{Hg!K`&_DOSM6O}7D;B#nX{&6g0=Uqdg}hvbAJ?kwlV_#Bqfj(0aO?r?&*~9#2KbPTDNJ297F@^(xo@BJz>qV^5wB3`9hA2jkVNZOb;`ut<&OuO4$^7!}9p!K*K%nWhwBPSc)d1QDuIbB}-0!Y#mtI*#$3-;6`AYg>J2s<FuIym}grVPc~2~E!LLz2A~Ng#3y6VN`a3%`O)7eJKpTJ<FVS=$LN;J;WsZW*Zz8Cuvevas&Cv@D60DS2C{T#*6SD1dVBNP_3G+98?Gu=j^s)ycXI3nh-Xs1@A|_Vo#UzPi;Rt-zuxXx$M|>)4eY`Pj+kJ7-CCo5%t@K>SmP|+U+e~m8B^s;R!Mo3P8^z}E$MTjQoCi0<grrK&QPQiQCx=+#}!Aukv&x6locSr%jM9|=#F>$<8(!T_1?MLR^1apg#;Rr<qjE$KpY3J#2WG27@wu_H&SNEYesAm9P5TqD`TJ^lDjCuMam5T9TT#klqMKZNOWAb8MDVJ26Rg9_j*g)@D|Z=gEVwmA8Y%%`qR*m7)=_=5|_CE<o^a65ahGc=0+CPRV4;24?*RE!#hX>1Jq&xx}g9u5+Khx`q6mt9>)4qwUD4w70sn~eMmVXWU7fui}YAp**KfxN9aa%V>06Irh5dNjf8vCqhU%GrsHpx`vOHi7>>p;EHYkVys;Gb?{0Co%bD1rlopCjL*6mpYq9Tjxr5&8!Cq&gSS)2}ik$MX%&aqdP^cgf7M$J8u0tHgA{pfjkkR%IkLJPx1t_&y0Vk4)1BFR6VDvYsL+*ow7gdCdp@Q@T>116a&hme}b*|k=R#PbJx<A*D-eQlhE1~U1`Q2?TTia{E49vq7b%zhbe%@n~Z~B1Piz)+2Eo&tnW~~R!vjB-~X0@O0)^43;KL$1_WMi%RhtFi*wBj*iHn|5RN3pd<?k%#HZ0=|zJ=FY)h{&BIS$MeJ=LyBrucHc|DXQ`pP-%U=(Ke5%buiNnN1<<X(=fFQ;qh&R9#(5nuNQ^K_?k|Ew7}Nw=5)(TB-qFThqdfmXI9Y1WbFcapnQL>appUq<%qCHf0RcbZ9ad@+I>uhcH&j_6x3U;EuGk00~kLV)^yC3%;qPi3hp?FyrJ~IewUAM>5wm<ucVn%42Q=(=?;dd2xWy!oRj%Yw;?q@3+(L^Bq3sE?W8u1__FM^nUxUS)qM>{kXS!v_-MDLw2dtF&bV(aKDhAycgY0)FO^E*&CwKqCECv~={IwyQO!lQk{ZjoA~*@0BB1f!Bnxr8+s3s6T5PEn-g<5=soqOrY-lKkYEhxoBPh@pjp65`mGbzzktSCHin&T$00$2qXyd~AoJ-p@uv)i}uTrqwcRrA@Ky@qk!Sl^u-)7d=3}#@FZ9v;KqAuK4G^X1$;cCEf?0@=s^ADTEX||BC*%%P~AaotDec*GnSsOoZ&8YQ2U@dGK3jkBIJce{vj;sgWx8d=zyys}pQ{MFkaecZxOoUAvE&J&Iw#^a)*(WJR0fdcItVCrv+7c|H8@Gdog?#XA=6rY;3ol%OGsNGRx+CNd&8SWTrC;%)s;8Za@==4{n=2j7QLN|~){XkDI@1Sx`aMRSOqIkI$@1j5XlHB(Jdw%g)1QQP{PvmZ(sQ$bXM64#?~d=DT+|VlTiks~#eNrH3<}o-Qzfn?VG74GCL=wgs>84%!>=SzlQ+{JQ2uN;eRV-Nu7wglM%4i>83&kJpaO19#(U6*CP!4&8(02^<pK~Wf<7su{8uCBI9>)*qx#UeFpmyuj0al23BxG?B>MXfYd|$4mfSpzfTyfInd@?!bpg3;EMRHsLepaN)3F64?Sb$`CbP^;);ybv>aP}j%4ozk!0*jZM|VaJ?OK@TBL?gyw)N-yY-B#RAd<HvbGwoqF|ESR4layE=cCF_GM^3FDU8Z}vv~-v9*W6Cm31L~saw_}luzG1?qY&{_nGRljqKjR{X_lf(X@j{kl1mx4FxNX9}e`4rhA<~^^Lyg>`f1h3=U(i85ddg|H-|OO8SN5?zgA~G~GMzxR$}9rVE$8c`qWjsBe#ZYg(=`QQQATuiX#*9O+G!v#&oK69%3wp*$)S5b^M7!-_G2yFDEGXm-`T{<VW2joC`_D2!rFAyfanz4Xph01WqXFIg^kZSG>mx?(l4DtaZijkj_4;}KBjW0FpUHd|PPkf|ug=)+Rb0P0kV4lYwF(tPXg(i-<H7^3H(CTLKONxX=GdT1A3Iw3_toJpuXrdj?|N%};H)NO=?Kg})bg^AiRcFs$Mt)^uIJ-`Z^3yoISE7!I@yR^B!Uc2^0_?F2!VY3q~RB%XC2iUielP*V%)ra+slk~oN^Yhx}m%10rkz66Wt2kGwEnlUn2~}V+!#dp&7*`~`H{IXwj2zg%zi0S;;g2A)<^<P4Wajf<@Ocg}7_x<iN8(CdySnnPDQhEu;r2*)AEys-iPUI-C<w);urr4b^ruI<gTy=$ET&?vv?kE82Y~oh<T<YJL1(_{w9r3x-kUVV_)m0ONrfiUK>8_w<B{DkNLjx(sA^ZUgKoxfNwCe0mtT){c42xr$e+zj2SNO`*<QDu)|d}if}g{9z;9phf-yW4<HTh6h<F=cLWj@Gm)Ha9;AsoS8wx*66boeOCkE$Q8pvp!_#)*Z0N)WO;a)DA5*QVJO=j{1_>)i=0Kwrz6Id;3oI*PY*&@Yn)6{g>#v7y-Z{pn*b!EI!FRHgyH*u@AHV?*M^_`x+ejqTZ1HS5{E%Fz}=9)C+QyetCIXrHEeqSAZ?<rC`wsWV~)1K=ZUiQC9g|TpWQS?q)wm4qg?pK=dQt~AsH6{CYwzsY)i5*`whNe)q+gFd;Pw$+I_=Rr4#P@U`6^}_CM(@mIu9O2DpTa14FbpfRGleowc&1-;50#{2ym-$K@{@GTPx&(qv9X(uD8@t?Vojiq>e5(PyyBG>Z!H;z2?%ifm5jsQ=;PceQGM5T5UJN%1)Z(qj}`!05r2;~3woV@G?uK{+=}~9JO%KY2EVD<F$Xi9ZlR!AM0i9mJGhmhYfMn{gP%%oR@vd@Y|0MUmQXZote_1@(^WHDP*)0{QoJJ^jOo%pI>MKVu`p3vo&7zd!+rm1)^ZDTfLa|x5aN$?#CFH@IMIYVR1%~8>CyKFdq;dBIyvt1Sd+6j<v(fs%ai;XsUpK+QUDjQEHgLm&!JW(Igc!>U?K2vpV#56iF>C~GI^KiU~I<8&Cl`(JMlqXpX3lE`iD^isq^Fk2R|$^I;cYqQa?-r3BQ^czb|KzotEzcDo-5M3bpG{G?5rS{vpX0w|a>H%8|5*#ZsB=Of_&KjiOsEkR$he&?jAm(YA~U>pMaqqG0i}Z`~}_J@N(Wggd+5erW=ZR{Ljb0PhX4R=<7=G$E+j&CdW<nZb6Nj>Vj(*Jz042j*niBx0`%9;nkpXez7Z0jL$fbyl2>&f3`vjgv2GAAj6<bicZM4NV(w-ozPIoIkp<E#Ikr^&>|Y<Fu~fs2#%*HgZPNK(R@Ri_jb&Yh~w3L4kyj5(6rvx)L!-42!=p_f)H2F{UUTqILzX+(df7eUi<|z?kCEu&|LY1!0d7e^3hq*Ib54cPl;b4Pl55Dl-|jpM~!-{Gu={2_w!bL1$+;q!2WKvH&WHo(XS+o!*|&9&~JqS2Cse;Xj9ca3#prVkC;>PN*l=K;UNP;BNX`?m@gVEyL8Zv1X=3@mR^KL4k{fk9A}3(4GNGm{$|>d>T<p!z(c@yE7d1$9Tp|!!$8L4gdockzce7Ygs9~zS5DflbM5ypd_T6;h2h^8g`u7g^A3>k(}eelStN@aDOJ<pB?Vc-R{rMuq_R<H6OZ+x>=r^oXpS13&mp?nVWkkc3YUYKTPaA23kS4Z~QZzwdHQdRnKIz%r?UoA%7K=6R9o7F+<y_)l#k?BKZ_~J7UU)iMdi4-g%QYNXR35hx>*`o5JQI{9p&DMFw`agtnLDDQ7OdcI9#H!fN&WDIo~F^gYMJ0nS~k<Ky6^@4H(>oXtXJZf4?$l(SLH+E{>W&c=Rk%Q(}jB2-}Hm=)veR%gzv61<-k#+3OsO_q0D#NX}qMPDHi7`*ogHAR=Yy;6(CF&c%8%a||Bm5eCXlCciILFM9evlLg}6e9+S5QWK{Ux+A{Ej+TXZ)gahONCQdf)hPaRS6u~)v0zWCAbX<x9bKzUdY`2aIh3-v}a_WJ#Y;-C2Yw8ZjIps0|R{n@A~5-yE{&V4s+$nnk?8zV;z<5BEV1208w6ED8u;SXp!$rEWB$jOevT7=tbehF5cd7TTyH9f@BKb#LVsxsS@^_%g}xwZ!@&+$BlJf6bg8<FhC`IY{$MG?v~y$wF9>FV(2)H%%UbIT?5|M9t%;rHuQMl)ao0I`Jz$2Ny3?MOy>1Xh36`hyWR}PIMouqNz%<yoU6>vY2J=9#k;V1!q8PIE(1f(=1QfM2-Jyp{m5wV;DOQD#IdY*Ep+(^U^3woh~9?t9&bR=pbU~SlRE~aX1Pq%v>EeKw5oSthz2T_JhpSEFEM><A-OT!i-3q>UyLniC^FE2c(yokBqbI`8>kU9ls95Fr@*2~z8>*tej+zp@rNHaA&@f+T1P&-SG)B?<MGMr`q$OZzNs$X-~iv7YnO-mdaF-=;($JAUrA{W>Kh-4Yv9F?)xTWW+}PNBaix0d$>zpYRvdX<WVT%-aUo6R3x(Ok5q+-A@W-uqXUf{jV&3C$;hRXk63rv73GU0?32!D!npJIojB?5J{4A_~HW(}p%+n1rHg?f-mJI9>g$80}cdrTb%exB-qOLKEX0=&kLp4`10@3X;V#R0}9K0R)gvQ@6kp%{l*m#K7YklJZ`~o%tFcc#9U5wm;^Vm9bq4DU7?iH=D6{B|_l5&yLBHz*}Mr(^991KQ%%jOoLDQ<If?HtA^yz|^{<|D5Z{VK6sQQd|>8^6Gvj`Xv9bSY~2hLa$mq-U`IAYHPAS>;4JGn64VY(%A(z3Cksfa|jDx1r6A`!Im&=~LC!XZ6cJ!r5tjc?wQh?c)9Ft#8mK3O@b)JVq6vKqP#EYIoMEuTE}0I|mDZ=FaP1ZEjqrtaixu!E|h{ea8X!@vI3;=1jTL=7!p?<mM}d{2}?KT%4H7Rocj16GKueugnn&v@iezrzGHQZ6l+@eS1fpgX!UszQKV|3OzgMO*-lH|Iz78?01Ix4~#fn@S`Kq(UnYeB%FiEaN9fm>GY`My5eB=w$51S%~&Ye0snQyLLCYC4_;v!`rYuZ6P`QYjJAzHCE_=!PFFG+3w2?guH<g~r!(5NcW~gHzTy2^p#&C6V4(ySN?@U${=T;<wt@!jEiCvJ7JLf}zC{J!OYhyM7uka!x}t482Y~e*c8ChYFX54)cfzp{d7I<kKaP&}0aYb9<aheG*l{Y>9TGygayxp)OwUa#Kl=FWf`VcxR<2YgMG<PiQsmz%l{^+%Z0p%OdH`t0p)__AmnZ2Y61$x}9f`Jq!To&$zHbTqPp;}v&-?v@J-up#$9FX8^A709jvYHbnCT%#ts5AnnZua{Sjvvw9eZ{sIy?TKfBQFd_5EO`7Z7bXtZzr>u0(QIvI~B8?|!p8vHOFWe%P1kY##2$ax)0B7~)hHQb#h>vAa8w?C#j}!3_1~gBiJ_7s?;ZaL5qg4q&Ux9gWO_R*t;xkQSMl73B{vOq)3YVj<f55?-J?b8hCSLlz!dP<wpDNX*zG)h3_ka-8C8!h2azK4ZBLW{?RwT*~FL3nI8lghBvByFZw5&1W&`aUw?lH_q>cV;RK@-~v=_EYaO5N_Z@26DW1&FK&IUx(DD`KGr?%78h@w1eBzNrEaheStEu$9FK7~_QaCoQGfMhv+Bu2EU5teiwS@%q9u&qyWP=7<jxja6T^<@PXZ6~7d7DleF(X}hSUvuM-lEQ246SU7g|^uvLqGpbm36h19dDhzRM36-bO=a)i-RFlh(?{<lz6J?HxeI0d;yLQ-M<(l80CHSA{vnWV~+Yp-DA5IW{q2ZWVpf<}w@@)ZC=<t=fgjae&M}JH`#LvM2)0RPY!mO9ZGbj|QgPAn_5r+rg&;IgTAL=t`MF`NIcK^$2_v`D#NB4wDNCUw`<REzVFLP?CLoFvFfU_#*Ma413`t)kUSehtG0A%VGhT#1W2~{7Uvh_<6ccnlXkR3!#^1S?i<b?3pq^F$gtk=}w5V#|`h`*c$bRXR9wyagfZ-we=ms$21Bz$&P?a=Kn~X{@dUE?*9g~PQ&<b%>Lh){a3;4$@rfC#_i2<yC=K(7^R8EZT=sF^nXPRAK3F>6Xgr0;lEM-e*((a0~PYFp`Z@s?eIPpg17j`5g8aFT=Bomy+8v6^w%Egi0|2%?1=v2i-I900U{v5ot@GD(=Lu!DD4z7)7i{_<r^gJsPTBNkezaPI<2b}h`AG?0g1MLF~M)G;(z`%#Yy};Cr-jdbtq4(|Fnu6k$Bo@i|!DoX@?9<({1cjdzmqSFdi2zz-@SRhiCJ;tBc9ZMF_<ZS>QH<Xszbi6z<^0ti@sa=)jmOqALB@xZzz~n~c%_jR)QOzvxEC;k0z88~ERUQOH)Jzu@*?Y;)M=UszK(GDoK9q?RsIl_VWCA5xVoQhgL5gW$6#h)|m&M2UR^Bcnb2{pnsPLk5=e!`=hK2h+~pp1tp-QNp+=bB!`!_Q!D(Cw1#`^W$OMK;>|E$R@8uNSv3wH(#EBaS!3#sQB2`A6FWeU$6%g%2BJQPi>w2W^3tEb?LLMrL#1VA|=O_zcn6xR9!xeIVc`p-#c=U6<W2+&#D){;B?(uWNhsHG0gSCsog^;TKB8V3uIjbU)oe9(pPxXqU#>CG%=W<3Z1<sij>j|!{%jkZrJbY%6E+iU)DaqB%*yk{i(jOQa$}y?coM$4X!+{KD^er`%UBJ%j)wLJ;Ej@PWpF@eR838_Ce$B`OS@u+D9v@ur@?QR&Gi$>_80l3Guz6Ppsni6yboOu@DAUj)f>nQq(_C5rs*Wk2Y=a?flH(tOlfzn*`V;jF$<U$vRbWEEA0de59&|!q_{C;w(44AN|m*-p<$n+=>LM6-d4LqfG#{Yk)ZB&_SFR5pjriE^NY;4=I)5h3aW?;e%EG;ywmdU0cKV&#rHM_7&!jd-Qa3eZBhhK5C2}VK5?bLYUTn2hdr?4a5$sUBCIa+Qs|z=T{q#K2hQ0WwGAeB*J|9*T$<ewQCPp!?bnxN%hAMt6%+yndxqPB+-5K)cxwpm!4DNIKX;T97h}!W=M1$CcKUdyXTbnY<?umvs6geu#9;vnyDZjHNv%4tTHAL$@rLps(aJq2rrh&tgp<+ngGHshH7k9CF^Oid{9Y7`W}NxM7@oU)$v)<mH1nyG~_o)jf5A?QO!_{8MwLd2a_B41AT{}$@rmSb|LWSXsakKa<ZI-9qUu)J##oCwSF`x$2l~YFH``s;{XQM9~U*#g4_gJM9iX<ADtRX5f{OprQ9LJNXh8r56_7B<8VIJjKUlsn6P}Ox)C+ICnph?*(f*W&aY=g+1w1}pg=a2FCRVWsMvTZsb_2RC$mH(gAOn=ZWb7AVbIi(XB2@Li*89KE>;d_=~$*P72%Q=n?YegX<gNxGTJ~yrFlRA3MO2gDHH-iaat`Og-sz$B@P;a@srBK;OU~)G#BJ%QFYAwjo-#OFk&REcr)DeifIc6bkVnbW5jhxkQlWiIv#ih&@^C3zVg`aaWBr7e4}rJ{04SNqC2!Z5l_TI$?i~BJh9kX)>+&?we?F9OJ76-nis2EmrqnzZ`98&!=o+REcIYg98>2=rhJ6)i8`p=m~;{)Ao22%Ovj!sI$^liWxmPWVKhC~5f%{(aWcfpMT8kvG#WpWo6qJC=gJk4`BH91cQY@#bP5HH7C4(Dn8!jY+`9}bid5<Z&K>;1NY~wF%rw{JAdb<0OXO2Y23`@JT8mh$hmBjoemJ?OYUw0KF!oD=oGbDKBf<8bN&iMRs1gIsL|Pa+T;5unp3rAM2Xs^;7r78Hzf&DDY%{M(A$K@4vA`^tdSagSKYJ0lUu6qQHn8D%B)HG5gxC*k#k_gVW}@CWHpoN@P|+x!_GBx$Zy!`|-sD)%Xr#09W%a8I)#WGEOF&sK)j#?g^90dXvduX!h$!M28m}%_Z+^bHep(&=>c^O*C+ss<a<zwuZM8ixoteo`;<2@J4P^2q{HU}Qi>chH1&wlvsSV}9IVl?=@+6KEJJ&%&Z4O;d=FbYG$o)K5&SjlROw`=L1<a%ll}g@GRrJ%E@||BxO|sVHi=&(hZ*H90x4V5`XYK0a>b=kF-+#snb1(^vT<y_?F5vABCQ`c-!0{YRrn-=J{`ebD?$`3{r0Yb(OXoFwsjhy$_1UTFrSF;9-n#otFjXgCU=G*Sbs&rD>yH{wZ`Z%xK*OJ?@Hfe3W-Gap3ep~tVS!_zDYz+#9tU(i>vDE55ZrAKuVFSR;PM`AG+z8ryKs*KTylyhU`f&H5*WRUpKe|M0y8d9LIwDd9_^_uU){R*QSI;N*c@M@#x)PmlpY>=O2nTR5#{qFGYJkfuLn)l>iF|I#at&#jEtPpOG|o!g;jccfq7_n&p=TMhHuXg<8!R>6xY6Z>ijkY^Uis_U}ld3LtOuA9izd&T&mq#uije6WL{SvtK6Kh4!IeOY@BP?Q*+eLovtlk+Fbi;bL~4`MQQn}t_!%_;YR9)`^cJEhso1oK?Z8jXjUvw@y1~WWBiNm?|`Y3?(5xpnkUC)l2nosf&>JaHB(6_tlM#lF|+92s4Vi8(3(f-x)5c<Ms^cb+)Z(!Rk3F|!T6e@T?C~@0b@VTLBMSpk33`>Q2`;5gUVs4jB($I^%;ZDEb{bNMl))FU7|B@Y<-Eh!`gSX2QNf=xV5hU71i$C-TLl4CP{gI2RP$IdlJp1$!^+W4K4OT1gTC7Ijrr6F}Veh^!1o(vh*+&J{peMMaHS4GE<j)Qx-}Lc0rAaK<qJ*dxSP$O#L_3Qc5UIG1X!ecOyfT>KQMEIKcvF)BI1r4Y4f4NQL+GCs-kv`orWbDHII<e0~GB0zLD*wjV7PfRBg@%zoxO6plTATwlGWua-7dvUh^E6Wudf(IZ*#yvj=0xU8ae!jt+1B@%Q=#F-rTl8g*4=Id3=+|5j!dsAFZvvVFsutzZ_HW0PZLb@bM#DZc$DJ8+!y8Ss!@78CRs!PwqL78h9xC;EPdY>3a3TzDaJp4i*L-i%eUy^tn664JSa>I$k!v&ZB?Mt;gXR0fo)GvQgUAaz)*vTp3)QQIG+ScvQwl=PBzFLL*Q;dUBB!=Y$pya%0TmHjcl$hSUS-o_DNjp5+8r6r}YULr^5lsGZL(nc@|E+tst6yEMp8jk7;Z?Ye%^V?2oFdaY8~<~zR4hj#Z^c5L(O4*p;t>ukAJvMEb$5<)aT)$zT!e4v_grxGJurjglG+~o7km;|vD`5U&kmBYFkSkbWI+P`pdj9ZbVLq04M#;6wJSSQaQCr=pW${a{J><sgwfrOVe?7iM>vrY4B+?jH|!hY>JoBusU~6jC;;08bSILY=`0lYuUtoRopuOga;DfZ6U_JLsu%uRJ#nk{@LuE5{o3ujn;TE`{Zo5%x$*KOXT{oFyI;HWy!v=$a~%!#tIwWPm#<*}Huk50&30-TC^$6vKYL;8_C1v3l2$@)6{R0iW5PHzcu#ilpxwMrQ?P&Zq)(XJusftOKj9-G1^8=vC5S9NAvY9KP>K(9z2T0e1RB!)qV=t)XT5{UGUjf|y(&~15N|e7_c<C9c?$WDmNAw+gcXiWjTzX_p8PPzspQ5a5)LS(2koBDSI{+|a{xOMpQgrK%o+g1zr|DXl@@Dneio{r%R43XN=2qZe}HABII+C!e}H9H5vfSZRwy{61k5^xVsSPyh5D5o3WD-~Hg!X_o3CIiZ+!`vd42Qc&HBb?Th|`eFFs->g1^?*FQMKNc!t`>-NuU-?CXWR#Qm$)kADQbRp0na?e-VE$2BejUA|!}`c!`GJ&3+pu(9ywiWDicH|(28G)5QRl=!Q4TU`o%6{0uLj<1BcKD1`KHA;Xornkd{&1E?bKPHhs`J%e^b?x^3`tlRuxc2-n)rZ%aw&yq1SHRzGY;IiKx^Wq}Id;L>I&%S8%+<4vFHh;@K%nBYl~Hx;H?qGZ3{6ev_A&~-m3Rnv<-NRCT|R^Bd}|$eMAXx~SjGGY*M9=s#cIID-TRwsCpOo9s-1o86GyaZ`4BP{r+Dapshl=wZY6&^(<JM6^S+6)lyg(2_!OoS0@^`-IfCze_1p}dnmx;GZbDT~50CZ@4Gxbwd*ACBem9MIQ_}=xP@J(5*~^r-lkpLIB4gBd&%uQj@$wQHkIdUw=1_s1D`+@vD~m)blyZ=4%%F(9yIhiVawDXx+Dpf}A5OLGV#GTfZExctr`6^YE`$>=Vev2HMm`G+>sc0vFbBH1aSE_Nb#=LR;?vsIhmCKRs-K>RtzBP#)p+zVywOf~fYxp&iqYZE&Z%SgE9whNG#Qsx0$4B4IlH7SMsQ&<llZ3zL>5ToDjt7HqfnuUoHgKr+|1E@i3#yY_(0Eod9eBi-*wUl(*vU;VdgcHVLvnhEO!*0`%&jM9#{rrgrI<Jn_{^2J!&UT^g!O)aW&808E!L{L#8RjwV+H)#4`U{tI%|U|Kp~FXb#a)hYnH-Hv*2Zgb83Al~onEfZX(KWx>hjFl5La(glfF^DL<9rlpF0P@Hg6rIJs?<p;DwI!w7l@nvg`(6L9+mda<Z;KG%cE#zz;S_3_qpGR>j;Ta^xN-X+BxMJ#nFepwi8K%bxRQ5->wqm<k(_LEk1UD72YBLC`xj`L$n~04{zhf%DF>dbMe5I5LP&8;K*)RX_Z$*ECI@W-ZEhZI7c{)H51LQ*-6F=Lvj?9*urDXzLT{ecgY7)1oQuIiw2|H*lOS8xzrmE@*oZXp<+GklTjz=G$@%BNupgTbKWhUwXLeoVF3<}?I9&K|+pglmAq)|qBE*iJhfF8;Qev|U!A-H>Wm;psllOv!+nNa$3E=Wt;BM*(Bl=#-{>je@Qc>S$2D@ZQU<;e8$=JOj{OE(&?&T!V^+Qo0{H?MA8{;cunkw>GC$4*X_2j2+^Su8YL$e>lCRI_fAmfR<?H6kz}D2km#Fzq^X)hBsAsBk==wP(OYaSRnuwx>FI7JsMz{l0;BIbBmpfU+J1S_C1<>kdS0%@zdsU|_ZDpKYLs^wab3vT^r3i`#8_b4{hEhD^PXN<Q~(wAG!VQ&m9mV}dd64zTE7-9QO1X#y;E9cpR3r$r^uwpw`TQ~w+L)Kg$@v4k?|q#BxirUPdA_UAmOJigkk@5JDi9|OsVL%j9aIzA1p>`LZ#b%lPP5<Svzmisa=Gk55X5LSP!K2ulgm;Vl@wDIKQ`jzwHXmhdxjl694Wmt&xO>~M$zAI?yN{gj>`>P;Ub*;IZfzFW+J-*I_RBr;ge}j3t>hkUCgLBpMS5Zr~etC0kz5eaG5O}WKseXJ%2stSo9^*30${x!h8AKH7(QXz?d+?Nd&dpgGJ34OHyZ9}N(0re$km-nMuSsJ0Qs$WG4RHrMWfxK4(k@`N)T~PvX`>^lH{882o6r!Bp~XQIOK>OQAUOV=7E@Q3yg@6GT>^4wm(~SriKQP$qn1IdE>JUL-8A%FG-ECdchbfK(ktuD`E^(UC&o`lhEYW9KKl1I3KG$<-ad|C&5*aK<2otnea~mdo7^Hf(&UB3Ao#q8$;Sh|Hc-SFSXCTf6b(5A>>rTqw5uaIysVD~U6VGJi30ECn$_TG3SNfX!O_8f=tmjhkw5{;w>59q;OmTFRl%k+283)LhF_UD;xOe`j<muASgBASBn#U7ZK(DW3d4c%m<2SPKbi~m|M8#CZ(x|ITjzw~`toDE`EPuL))h;i5o)_85ZQ^RPz>&>+U@)HT<av!73W(-37%}CZ9E#3E;{JwXO3p_1tHCm@!Yi>kgX7PB#Gz<qxLkqEzgI#?f_z}ueU#~!s_MXN;FSIj)8@8-E<g;>}x`-sIl4Df=qTnqG8O`_cQvm?O0!D7GLAGk@l+?W4Q^Mrkw*|W1(F%TK*Y`q4UC_bBy7SBgK=jBPloOb~8R#1b0K315?&9(e(h-hk>Im)Xtx+Ui?B(%<B&}UtGXAoM*q}_)m@H@9HZL8-M$v@d8Z|I7M;o@-s{lbm0nGuid&({rIl;3>jxhjgKcV5%av>MVFXH>q&Ybh*{Og$%27qququoru`=Ou({oWcw39~OPLaF!_|0<!SgHuYmQ=!I?{m5QVZy)!o(bfnwF&^A`vckZ`@+)(4x%9{yza-EM!wd3O4mYIT*?5#2|L+B)$-nq0RN&GzCFKBA+tKOikOD)ZiBcd`pBpogXz<pM>bZ6}7(742*|ttSt)t>8+)>v90{PJr^qFra6PN^hZ$B)UHnxoFP<Tz;(Sd)Rl-^00}{)tPZoM0kicE{+$7m$CXo}M;<*`O6zmJt4TTQd@~Q~rw_J4)pGyB6cvx=_<(K8w|ND;VG98pSj`mXa#h<hwFb?dS%zepGy{{f&@-IgKQ#Kj)0-aZ86FLLuv|*IN63Xu$sbk;D-m`kHa!A2Ss^EQFe%Qqkz7HN`d=WYz{_14@QGoC|JPi(!u`mhl_1BFeoa>sb<oPnPSbw?{Z}!Y^1Tl3R#ClWd%-NpEORg06?MJSEcUAEG=<T+s(b`y=;r{DCXxt9&a@SdDG^s9znC@(W^SUvem3So@Ty?K0qkVZSr|zVAMD$kc6x?~`wphv)8KX}<^J^d%Ff3$J+wc!r+jM7entZ>C1YR40ZnE~k)X2r`LL?sl(g1D99Rk9e0Drgh-iogn@ENYsj$E6z^;)9uy*b~nzfQyd1Liu?dl7(9bm!Xg?nfcb?G7-O0ZusOAwihL(8TI=eCx<s9!u|qcnBWk+tRM_OFPCX(G&6JECjTSt_s%wGi<Wi&pC658FyBw<EO30Lx08SP?p(P!pw7+yJJCg`JQ^tLx<&n#vq4ha*rk$>#)lJ3C4VVEE)HJ<OZ!Dv7ove6lG=e`eoky<fLqoF!D#k!s-_UROVAQDqWytTdCM--Y&&FPeo}DISwLaZ{TaZB57)dLu+qhG>)1|4T59%EWaLWo_8}8=<g6ydjATTM@qCt)`1#T5N>ab82K;QG{B38gHu&$V0Mi8~)uN(2mtC!kYC`LlV3d8A${g2E`qUNbNWjg*l**$es{j{)BD=0TtVVjM|hXn#`nQ1`y#CnV~7M`IHDIAOeXh-sZ-F53<|YKiG@+ayXq3c01{$=7Shu7>ObZ7v`7_k3v$bydE-~4YPtgHcxbu2tbL2ILUKNPqdnWeW@~%9MeYIQbl8lh=V(-EbQ9UHkGsr4+I^3X^*xzUoITiPL1kIV&3&f3+pCX(kI8QU&v49CKe_Nfz@@a9!6&ot+G4-#DZbawQYc?1Ea}I36Q!c_&VJNj3}hR5XY=|%#1%n(QY^uXoz>FJry*x=4PD%HeV!}L9LXTpVg8B0*lL_=>rFy{XGMH@1#dYMN){6f1OdXg6Xz7!AVpm-^m@#7w5{Dnu~4=4LXeikr+Ox<XD)dNI70;wcWH#y3y>&+q>D+l?_JsjvN$0^fsU0sGa>w_4yJyC{hd%G-*Gz%*N~(7-aSDEVAERyH~q-w!ZPOvHGxj7Q=&VK1WxK+W9lJI|e~$bVr2VH{QAN_tki$ySq~Ocw+A(I;Sy<2{TeVINPmvn@adzSTM74dnX%DS-)o$UE;1d0p&e;wsm7)C}7^w@B+@dblA<pA*t|%ht4f*f?T_`RDHa{hTOHY-&TLT@=sWk%g@-r``?#N)W2N^3cb4Wadr7|<K;`%?!zjf;(zH<_2D%DRX@ATw!dLZ^LA9T?anl5Br``eLXMh;Ef|my_K6c&xMfo+Uqo2%VT|qr)P88Gzcssfts1=yYervUzS-74CBrFb@nZN30xahMtx50TP87VaU1q!a>Y4TG$|r8DC=#<nx8OpWm_m9smDC*~n9fMyvNDGRHBqMj;xBMkl%pvyUI4v%`T{H{%JHw=VS8TTX+><==8Fqqyj7HEU$}mA6_se;e+G4-0I+DT)(v*g)|FREwYGKh3IDF$_z=#o<`M1sc8`UTpZk!)Uy)-KPBwZsk{%9F13tqbFh$I#=Nl(qG?uR6D`>j*<DJd5i)^4UGPoC=#nw)W>S(h-E(Z84=J4^$+SzCH3|n1uI|yqo89>D3C-W1+^Ls4pnm=5+=(=7Ji%Kn#El*!Bp9=QEr`~<LjxEv=K|j9-U&Hd@o7H=48%>q96I6Zn1Qtk{$nfOr9aFa(r%Qn$?ru*qH_TWR?XO?l68&914KHiAPXd}1n}%K2@BUCdajWs-Yt-m0UsX$t%f0+WTc6?~6wR;)vuN-u82mTYi>IqAKXoPcZ?0diKe)jD%WTlLb$7jX_T1*HRVc|jxc+>-`qhuHSzPI;WVHd4vekH4A0bARbM$mDDozxyr9}M~CyJ#KrLNq?m7tAe=fBjy_;&01-)m2w)-HY;_TU2`pYw-E0w7YqwR8`c0ZYu<YIrVa0UT`_jB7gT+TfG9a%6<oFbx!&o6g;pp;Bf}prMiC*As$td)$(sh9}Ba1gka84h8+9Ld1!@b6{XECMy|{>A$4ooC*S==V2#%qFh$yksqT}a+Ei<9QK%qa7HcfH6}=1qG8-lIAwXel94dosh<4mF45~=bKFGKnq1DG)mdg~$IRnpk3WnWVAFuT6pL3nhF~hx5-`KXX&S$UAv3;5{N~TB^Fp#pY>TnPxL{PYt3{5CW*QbI|7SFpao^*8mJ5$tK{q$QW=XkV7GN$Z$PBUy6mxVuv{UcK82snaCIx1)A~P?Fb%|<)U~#tc1q6yZPd=O8@D)0bV$6yb%A&1_WrKih^?|Z04sbf8w_vbb*WHmcZOSy9nhWlc6wV9FWqovU`OU6@P*aA>BSclRu7RcktSJ->=pK20a05;Z|6e68Gvkpul5Gr<Sb3)31%oPd1*rKN*o_&C&<xtZNXYevfKvj9q@#IbR`*<!CV-2pc4e6_9CqnuYIt+wRsGf%7zg#rt8k#;#~-}Z<kDT%+)_C{H=8Bt1t7{4{)3JN0^!nuwvnODD9|9;vWM&)Gf3WXg&tDkZ-7=uT{R+`V#p_Vva&l~q$L8+dQpF{G)v`rWB+=Q>un74yp1o0j^o5FvKP~F{C6=lJT%&sNTiM<kH15clL%2Ead)U1kRgHrPItU>axsK)TbD11if&SL{M_-o#c(X-%Wi3B%=Ii`ZU)wnC%vfLIpq`%A5Nm1SC1S)=72-MB(4W~M$^6FXaJpqgd2h@M<}{{nS}l^UIw`2W~NfQDwRr_bjTFN35j&X_@*Tw@(Sg97QTgcV^9G-#U1WFqI9S|ywrWk*@D1v5BbyB)VRUQy8R|eH@l~so2iAS4U&e+6B?z89__Y!5|BMkC&JVVuA=uE`9hD&9cFXJN^kd!r2G2@(qYd4c(k}1!a;KpraW;T{6dRl_#@@BsmbthZgJ6Z_8p&se~)W@hsS~A+-n$d0{7}vN_ups=-=(ogpMV&-x^LTry^s?j@_7SuXEfo))tJ3NGb16q8BhbS`xiz)GJ|yIWkW~c_Jf^g30@jxpO#_gxTv*lLgaupy%EFd!3QN1H*gMPCwA9j6_g%OL-%q3QcB*W&ZTpHRcaDi;PX13Y$%Yh0CR%Vly_Y?=<Puxtq;!+=CVHqAX2|X#fhBn5W?B=#ZOf;J7RdClWUocYe_}Asu3O1x<(ssmZj6uyV~NgHe;jKE@1~*&6YS2rLNSII^#IYPeT|0{3B0H{ik6(%IU%Z#h_6b>%7t#^$R>4}Hx!X=!O?WV92tn?KdpU!fz|t)<Nu7i(WG)xNl2|L~dS5RJ_na$m7B!02W@p}DYMtdsNfRS%6Ol#MReL=7}!RRb?50}nG~Z-i;;<yb_cN%$vIA{;%HYmT|Rppap6+>Maj9CQiA^zq8oHbN3k=1Ms>coeM@>4Q2M(kS?%5UCp5hB%lE<Ib<&_$Zy=HLk9FSU>fEWf^WfzUNoBmhK1-ne)$30t3Y3=eJ0zhWUswzbcUp{GP=b`dkX%hU1aa80BCmf=gqqVcNPwB^C*g`+^+7q84RlW=hOhTT95o!>hY8p0Z^k2};y&PA=Tx^xDcJj{PDm76V3{CPFm6dkAoQ{%{_6aHdR9=$OHHKVd#()tjH!F28h_K}#3B`QkKrBYn8JcCr5OCY;o8)W-B$%)_$zNinTuq|PO2=N!(=QgVcB{%AfsM=I?x!4F&P-AKAHV@#;cOn6c{UyMaqVI`LxAM}OytHn_)q3l%v63INaC17m=0vH+c+=t2$0ZgdQS^*Z?MIAwz7aDCR<1qq6?`#vUPp7|8_dA2u24{ToL)KPJ?kQ+tyB?^r=If}<??^1Wv16GZ(-)5JDDUdULg|DfcnzExJAsu>$}+F90C*+}u)l!2fVnId<|;WHK??p0XeHsegH8FSHiq5M3wW0$5NglpD1(C^0uH@G0QAbUYBJ8;bS{HTky<{a()YweJNbiWy#g#~wnHBj43=BuuulF_U~4%4j+#%LMqlS*Tu|$lG40kq`x}EjR4UHrr&0Jkjh;-Yh<_S>p9;m331V)nV5vZ&!vbE}<~Zcx<{ZLEQkl|1N?ijH?>#`V*nBTB&N(B&Oev4@qLh0OyeZ`I;!&JAEsSNo%RwdNrA%O2yph?<M5IvRi<!chp=Rsao!Vcoum>uWX{<G_U;f6-8Q-{jUtA`1Z)n%?Avk}yJvr3b6v<dzKb}%|2(M`xl(?rQX=8*~mlVwLj&6deiUFzWd(-K8v*^!auMcTFgtO2c8b}?dYCEVJiaL0wi?{TEl8HT&B=M-<#Oxf^r>nAnb?XaJhLCjBFP^Dy+_qE;I0&+K<JBjd&!00nB<uM$KlIopF`>z0A6h!!WnlcKGbuAAyh|pgC?e|=kVr?FNQ|+^mi#V594yAnSdppl=Gs}_1OK>&aTkfIK*U{tc(!)?B%M9Hr*MYS$zAj(5pAdIE+(F$Z|vEr3Tkr#<@F~hsA;z<c{rLV<V>N`5EXQ^tc{l^tB+ThkPMp)2=mrHu2=&pOfeijUu6o3EvYA(Y^l3j`L1^U?7j|YvU>T^=GxK#91x0V&7R)Sp@HgGS8Er)5&n>0KCVA_41{ds)urmkH<6ZIK3!e@o|hKt^3{cE#p{oNsi|))S5JMj`Qk5#`ejU45mL{zD6#S+jEP3l{b^E4>1>B)PQoz?<il@Je}ZeEc22Z;_(jGUJn4caU%QASJG}vq{2qsX={#bTbD!b~czl}Q@baM6`-dMk*UxWVe^OgTJtJnKxpcmMbG7>7O4uv<YdX4^yW_Qb6Xm0xYWsUe_rB+h>>V6VkC4F$(4WVryt}?6w>yV!gt-Z8Xgs{=auK{H<1Mr~)YCVtniN-V$&fggG-S4O=WO)#oW?v|VjTt#hNRf~0Eh`Ot*higH<|fD1`P`$B#~ey)T-jObK9t;ILvxqlTF4=c!UUW?Mp5!M0ISP<x?GtIaNS?)?2D@e0mCgM_{yQNG8aa+<d-Vaf(wKj-1Y*8rq{<J<bMM;qFj)bhxL_5G)f_ieFG%Y<zGB(H+3`FOJcWg*+H;PmZ=|sFTC*^z`+od&x+OO2_9Yr=d@kiy3gJ^KlxM;r}DnGd!}dZ)gaB?H$}d)Sn(rD|0f@xT>oBfL-k`-0%+H@VY+Q3z>TxZYH50?HSn@B$nsqQeq?Vz)UeQ)o^VleMQE629)FEALvuPG_5EsONDEaPwhj3-sVkfG3v<pE8*GVyrabS7UFGGhU3HoA6#C5A<pL}=15}>yJ^=Pn-)e!_73+AjYjd+E;gEw;z4?aeHA4GW}U0z(Glh7bcz`Mk}5iG?6pO&;)hfvRvH66go!>=)_g+WL>Wv>Qwg38+{cCDVN7dN1WGm%&V*5PIZ@1#c@^ipiG>c$AtjHml2Hs>$(k^CCl+U=kcDX*%8Qbc`Z9RTl}f4d+#w*9CeWI4WVCniz$k3FW7(83TTv36>T-C4sD{0#QbeJ_*1RW;is%O#;i0oMCAon1WGD_J&V%R+S*oj!4bd!!y?|I>K#aBpng%p82`FHo2Av`>wncy6!L&S#-iUMdt9JOnz(C)?yT%D->9lquh~qq)&!*(CFzB#dF<AiK3wZK!4vsw#W7@{SuqnKuVj<q^j<g<4;kaThuJo;;tJ1RwHv)|-FwYXEQ5K!Xc*JCvihHRm5q*sq0lE2!+-yZ(*inX)o}Yzo2B_*$wQn20rid34A$sK;DbAJ9e0pbSH;{|l1?-Z25c>FAKj6*Ucyg-o>de-?kI+AitP6oAZQMQ2nMjdb-8cp0XYJuVQK9<mNqyr%edA&E(na`J<MG6(Bl~g-hl-g}wr{2a=WcEmj-cK&o!}h)^0`QDcfRP1mE%+SLSgoBMBiQP*-(g&^u0Tp9^T(dCL&o~nNxU_<JNs*-5k3;o8$GDzs0e)H2!^k{r#`wiYdV$RKFvXmKu)MEEe)DZ%|;iHp}5kc63U*gWibTcY)N4i-QE2Vo~SM=TQNd&7~$NwVtleKZZ^vO_7U+Ak+oOgbdt!gw+<!`wyE-9ml2?(Jqt3<c&usY8NmrvvD%~H>6HA>C}>i3e!~n;)^h?myfPhy?j#!9)%45fR`EUKPU*tAnSUj2Mn4&+&ee`bfKx)oGZZ!l;_|^DlM=HydI;R18+B8Y;2ubQF^!9k7sI28)%<=;#Kw3h1#Q)#-oompHo7HJI}XHzNoHXR6ZePqNw@6-f~7?HcM9dRiyq74KoloW7seJEjH>@4p{+;_)J}63l1pZOpD>VZ`|n4$izWKVdZ5>ZrtsQ46c0~gSz8>#{jtWd6EQqDdsc?fS0dAAFKl0{t-4xfg)m_fyE#{`7vErg;uSoEhNem%j(8><87K6iK<6Kzt^of^gPA*JWr-6D1fWvW=T_-%yqe~5eNplAWG5{*b2-~$3hF!QV=~aMZA%PvdLY}O-Ca0a0=(Aqi}mg7U1W?G^?8D0Rlq(bAA>8zyoG@6AF>su4G5dQ+sEqgX>`>XwhS^Be$+=0$Og$rB1`iE}*gsH-1@~1c_kChK5R47JC%Llf?punNw=v02d=AW<oLt*~z&KlhE$&60q8R3}`fFEoCyJQ7;}*F&el=91{U3PIIS33xtxmfZ_{@U(r>eF2J!_SH7TL0|)~biSsr~_uROVW(X(P&;{+>$;e&IP3lmwScr&+s5VplK|R7J=2Q8-?8<hwdYOsJ9O{=~`@=QCtuWI^ufjAP;=OAFtz8{tMWZ4MdDO+^c!Yu&5qHd8D$#G`529irut9ejx-FL4!HC(saNRBin)vY^w@&=domkr7el&<?zY6TNuVW~ELJ(0Y-13K1DE=lK`JTO_2VkELrO^a`z@ze=E1*J{mJngHNCK{MR1)CFu!ejJMtVqP3W@gLZXZzhVov33Mc_QUX3*P$Ck#4!im)fTfJ0ijW)r{#e0W;dkT~2e(vq~&Gp!MuO~0|A{eMCQA|Yc-cWI1#>T&Yb9`_#{x~ha~m7qbN;n-=fosr&DIjg_dRG_K@x*F8mkEX5iJt=}r6bd57x91iuP;SzoDYP{34%yV<6w&rkidTLw#w$JrEsP)P+qkGS@35rCuA#X$_mMm4Tu<t_#l{GLUMV{L+LA=nnxHX(s2b%Q+46&)lBLN?vzZ|qq&sX`EoZ=ZX&X3C+sNpkTzi<S;!I(I0Wfa$f5j`!R}&Z5UW@Q%EXXQBz@%Ma?0avcEo{HRUZ;F4GwX~V)Vt!o#g5CT$wGeCRtn!+yQ>Z{&u0qBHvMEbx$ewU&!meF+e?=l$L|c~y_#%RZ1QH>*={#qi*|eU&P1iNxSj6$IPqF#<X9rgYS(eR`NDk$L~JqD#0OWi4dSx8`I`ycm;^aR$AOOR4$<*reNS#7CLWo}mU^ESR4*n|5ed6-kx(HZO#H))lRR$p@Y3rNA6*PkA@vOO^uIsaw-;%=p}vv9UZuA(M#!}@un89`c?>g-39x3pCg?)bWLS&Fl!u(y6l6h1YN?W65Y9Gr9=8T*z!P;3meILIZQ;p^*`yuJ{R;E?3p$Vj?y$))HA^r-bl~B`K1De}TNA>AO1X}M-l^Ws7%#}|+x1E@HXQ~x2iVGYLG?36Jx}33u~4%s9!qb3#p^eFFnQ!Pobt93YqOh)ZFl`QsV&jDcZf}}IYWzXf09NH?BCxr{JuCyzNL(|d5{ewT223OY{s*P{U?u!K#yBK$3dZ8#(NxJk&zev7O#a8SbY-D<5qKBBs8EqUh^6N!z&x+Er31tJ983vCK_M^I<f{~V9|sNhs~8`%i99Oi@*ztPiaelV8SaIf43;8iA(;3F|h5neWiKz*R+yFuH!n|dk||rlbf%=Rp~A^ZwIl$Bs0`jOFrAzIM^kz>DtTBY!RuxibwI0E%U`T^Eg99_0TY$N#>&4&9aye-y}||nQCF0$s8{LP}9{w6h^zD>^XS2v3Vf`z7~_wg!`gBD$qY39~ZXe6tVH8#G;}L%{IR(=5QBMZhNX%lUg6P-~YtPxeFs7cWnb$Er+ZSJi|U*2`oC0<UOMYB(WidHUYSm<Ejj&M-KFl$|`(Hf{<~4ze8?ELxX(-stU^|xC)_B-5MX3rtC&@Cn`mpwOB|I44Mgzb|RwQv%Bb`c%jL3i3Zof1zU8Hel}Cdcs8Gl2wtx@NTmd{Yo}1KZK~F(Qz*8D729924Ep^@P~5EYx-`Q)i$)D`yaG?F+hMU}1|?*ui~VGg>7-<#fS3(J32@U<#h^H8NO%jS)TF5o)PLbToD!1qIiIMIPVXep>trp+@&Tf>Xs#V3L|ke_&cD%>?MW^MRnDT&<QIU^MI>&&<fu#u3XX-PG}cl3%I-a=l|^B^23pfJ(BmIMN!QF;z6|K|v0}*;bfAP^gXxoswD~N|aUt_FfEC%|7DlAj!jwn(XVU`BD^78NSp+RDL9`LR9)Xx2P7uBN?(pD&ArDR!CrsLavU!A*{K*L8Fg@i|GJ&j8?r^?b0V+tm#ARcRh}-$0q$7py;*^O;_b$~)9CXvvObz9&pn>anJ#Zb(!L^10+=@~iAWBOkwx(^&U}9(>+?Dz7p}8!^{)Q=Z68v9GstovcFBS@Dj-i#?y{Ci9wfVUi1r~nHaL}i+?4PcPfL<bl+qA2KcH%r)<y4ENzhv$s^c%v=Gi+MT1en^YaC8EUd^U7@@};SU%akUL<d1U5$RPb2239E+j!HueEg3n+x+L2-Ayy*g$viNjc?v)#fQ$Gy0E$g1P#50YGuksa(EmOr4~QTBbJz!33aVN(60PJgP)&wRP<1G-I$nP(yAiKU&xSQyIOq&U?PH@i1u_;2AM3_GqJ<0;>29V*qG6VVXwIi9(j{@05P?m7_N3#2)VC`~hDuS>lWHwxW)>pzEQI5dIH#4etCfv}9g**yMPEV;hB1XOWTi8^Fp+_Mj3U8SJ1LtrWQ)Q)Djap2HoNp1(Am_0T>s;R;xRJ2rJ*PzUx9;Rapx(59a%Es-{~w1rwD{k04k!q1Hy<`CZ)Kq+m`=^Qd4mNKG@+1h%DXIfg#v<9m_$IMi?j_3>7EhPqcaA@j2QKqBw2z7yz_bhMfU-%tTJ#!P@SRYG$S3F-&lCVYPbxlyHH#^ga7;kB5y)#)B~Rkk%V8w}%!g6i?J^StG*@x^+DU7R8Av9rv*H3R|qyZvBACx4%2Tb@_w{=JxrG&9#@c<*RJ@#vaG@%iq+NpK@w;%`ycc`yz_=0Jc2{eD>hpeOgdStCI%S_xWNNB6azaG4_#S8D*cCs1s`DC@&ShBis|+KEr(D(S3H$V7H3;<sYjnUsK+p4Gar^{shlL<Iz*}@VdL&SpAA!!#I^DTL+p>>tc>)^I0c@rkN2+EhKsk_d7*ui?-m0rdFc*^e0St^tTPndHDDw#wPsio1wnm)NuuN$1K@niI8>AZ*cdj&zBmjx=|v~Ie6I`Q*N%6$Ea1eghft=>+{^>NLU$d9kM9f`iR4uYae4uR5?Nz_SVu;_00O_i_@^_7@bvD9yLzB*xV3Puhs@_tuU#Yn6lcn`_)(1l+l$~t<{J1jg#7R;BJ$aWb|7L&Wd58dC#_NI5wH(D8_V;A-otq4LW}s#ydCk|MJt`;dIZad`SQ6-u_r9Q3M)VyAa{iWR{>zX7L)dn1R!+2%XAhrrn9f&cw5G)3buR%x6<YIw<f94FK_Eq-+PpAa`N{u7OK%I3^NtS9@0F#|1R-r~FK|c&sb}7hL{ESZHFQSRz`xc(3;NbLcQhp)Fw58jtSD{GOj|EuCN-v-#?7b>*psS3D-6GZj5tHV4a_s4UFp-27~?S7vf#_(1RD(68S9{z#bL@5b;UKt6J!LbB!bSSE70_#mvpkgy3w=|8}7VNN(li!ckbsuJh3fdc;8k;HtGXl3CsM_9qgULvRzhS6<{r7oJqv`=ooBP9CR6KoSNj7?Ra=L*G{!+zUVY+H2u-CQL-pReqNin8~f5l6U}=)ub$4lR^&lM!snnwnV40Aw2%RO3trq-WamR$J3RUZR&2>HR(GM^N`H@sj0!l=B}8C4A<ns-MDkSxiEIu{bq1n>LNLeObC>9W<mSZwGq?&W5BPD7dckhi9A$3e?SDIdBSo8>(G>oK9}8-AAKQ3Yh%Q=QlbN``KHtcJ}+)%E~{VpX-7*SD)4XdIj)Z?ao?T8#)oLEwK|J)Y#Vnr)?nB+5TpGvZFoO)!vb4@91prc(Z+XM|)?Yz4J}@&Ow#?M@IL<1ByJllF9#X=`UT$PV`oMv;wHTdg;REv$M(l82g&LAbp{4NdHeD4#;%=4&z<+gXNA^pFL^f9m+2WCy*m3U~edmEsYGl(>^jf`cC^FM@Rd%&YS`KqJp;Ksh~)&pJM;3D<1-lg>^PJPSq~nudaN#xpuL3_FK&J^71>FzOAJX*}W8i;-&G~DWJkITj$GDTX&y{_D_FWfAF}mx?a0_roR5D@$`26>kTN6a5?q|&dys2oSnB4ofw$<-bZj7iJ`ps6oyhgf0czKjYp5FPgh}$_EeWHczmDbxAVHUu5#7I=5+7s-vd|uyUn%B3JAUH*j)Rr{@F{YUjOh}WA*Xo+Pdx|jY#*>*RTX|d17{(TT3u4PylD|*^}D!i$LDgZhhBy`I5_TU0<#)J;$bbjC*K|U5ZfL*Ri{^c6JH7N}ZHg9OI&>6~i|#uW_&<(xmWSphz{}Vrc1=>pWIkskO892N(8j(Y=9#y>uS;LG|XH+QYACi`1?^64|1cHk6yJ;Y+;wrQu6_=_B-NfBF*-3HF7aIhs~c^RcDl(M}Eh<C=R0+6Q}gO=3oj_MW}%Bkdz^^z8k|H98fv?p+E0sUT00i`B3G#>3^oeFf{o4Pj;6+Rv9nuPNHGeo~nmP!Wik0!$~Wryf=>oZMVHgYLkmKSQ`fLxcambV3i1j4`0rFL4ZG_`N5Qz#aAJ7u=WjF8R#f=#7`p>Q~Ogq1p3~Yn_R=x?nHBsjXgGMqlHN2eq@Wv}WnF^GNB$Bx8xh?%1A=1TNF5Z>smN!5!y*h3jtjo(?FJ2x;OS*oeZ|ojQP&!s^}Xsr%eNI6(E)YuFH;Bii7d2~1N@h=a>uYo{+TLXl_(-*E9REY9dp6~{pH=!ftH5}0IuUtj-vPhv`sjs>_9(8HzEfG}WNA|g9+3uqB0G&k2C@}{9<#9_F3lpKs3UsHf>PBowc)vGw@h<9XsbPk1%O=8$}ju#Fq#c{{s2ge=9@b?sU9DX21Gb4}VqRXJ=uzHASXjN+u?{8gsMJMq(w$T4ak(QWxLC*D<nOH_mzlq(~qNa0nZI0hv{p6d*iyyFc8MzyN&t|gthN7F-)^PL`Y7CPZZvjgA@k5~9Y9}7i;1adZpRn70{q7H28`pt*LKF#@U*>y<PrMg6lfvfOITK}_7heBY9#<cJUHfPSP8XkSK-8NXPidRL)KFmZ-R(&coBYKsz<oin$*U)x!vDiffJjJ2DA&W2S682Homrs`0r(Bh9q*#*g)3;czIEd=+yTt^;-XV*5BAWY9fmG1yho~^0H5*|#sopxPVK}!+>^JKV1OJ~q;~Ufn0SJ^*3S7Pfa_^jw+0<Hs>KW)6sB0q928dAzUUzo8ZW+Syn2RiBQGxCK0os}0Aa=?sa=0reZB&~8jr4U*RhqJ(b4vvKaIjF;vuF(RJ-;B5X{V6p@2t=Ug1S?`wZO)C-_Fgt5#h>D_;X5c9UG6{uCyZ<1}#0255xw;0B;640CYpN$unN)lWVa(LV?-OfsR-i?v%{)c$%8P<1=18!$85&Ch`w#3DCu@`4B~TD$Tk&bJD!fsMcTi&!(@ftQ~!jq*d|#VRk)=8IQ~aNu$j;F>|`6Iws_L;d@6wZGoPR3FP%fmtoikutvmeL|=rzRSvYIDkv%_qF4=0ChiK!Cx+>UHP*5@}t_L%Z-;Ov9A}uNGHTaNUM)1knSbE2w?LvoyVs@kWXhmr2C4b71i^nu+dB3H$H-c^aEr3`kfO;GUX$PNaz@-Sb&6Dq%YB<k84*S!oo3!+&-F{DHls`qYIly6h#Msx#`C0OGMF6?^jPftv<g3UBs(01JrFf78;GmLOq8wrDK^wEOZba-UIID{WwLf&%vu+`1wCVZv(kIF*TDbLE(64e>@cFNOW{X<831o`P@tevmgKz6lRYgd0y@g^@RG->aw&DDo!#5o}DXY3ZZ?#rNT}=%tngCa3WCP&6UgXwj-6wY`MF=JzLDji>1Tu$wa*K%{_Y(?U~ZQ=8wiZx)N{2;X>-#(}uhy6X2LHK$6_EnNpdYRWhZ+^gG^DIy{Gh^&$F#677jnewNcy@H(ZFFdxYTqcpfiDZ<N*kB~Z-bU9K>a7hc={N_zj1ju#1sf1L5TSw8PlM&5Eq>p;@5>kLHvhIb;m0~I9RAAb-tLS2fpHUsSv*NK`$iVI592-9a2CKX<7;iBTd{!zF2B;`I=Fk8#w1g2!0v={F6H}<EffdZm6)GvJ)wH2(4)cG8wJH;H*-Y~~*tD(U>|FDDM{^Uqax+JbVF@2t?0j}^dV0a=z5Lh|Dht7O318b??D8B7A@`f2h;}Z(j?Q=spp5=CneS**SQKlaX$f?)YPL0?Y?)LnRGa=_eQ9ls3J!RRbEmtT8AVe$Gl`ELQbWa_91BygDa;)|a{<i^Eg;+4w!t)-)NMVUMiU_RkQ&+q1qRlz3W(gOE_yC0H=+yK+A?5IG?##MCyQfRU&bRAZ4Ryn37)B-N8f3tuF_;GK4)v2OnGDhU4K!26-ny8=mJKX{=g>e_V~901#k#3N}N<Q4wUwWhxbAE%8iVjiW|P$<4vvPvt<_09?g|dK;(-%Hkn65ka8t+IG3`-m5zBbc%nnMC>_RoMi-<APKyYMfqz_6R}jS`qll7S$gr8gz*52S-93#<>wqRk0Tw#|Jm!F`a|KMCRZa;zOyFe4!ZY;B>qH}b1ZHQ8T44%LQl0}mk||H*^C=}*&7v83N7TAJx|-d#_tg4PK{`7=T{~c?e5OD~Bb1kJE;B8%u%(ku{~y`@pffS;u;Es@3w|C=hK=pv#=oI!%;z<ttPDmD^o_!B`+H=hvLFpM@6UJO%RsU64u)mabX=s}k`dH|jzm{{X-W~Bi9)lDZi~W!@goy@vN^))EUp%M>Bf^&$YW5xWgvKK55EK^FC4W@66u)@ia=?dcnSG-jv9JpzSvdtCU0)Zy$5=GoP&KMeQ)=to!&I2^&Sat7t`(0Q)Zul9L79?77!@fAc`EP`sOQEl2$L=!CY3Se%xIDzIN+E_2av!+#(nMcZUwRLnf885qAo8`t&vr_zn#+qV9-zlM>bE?KbX#G{F%)*H?P$_UFK}v)+suAxS(k>kqtU<LlG)n^$Ys9&jWFm2$J#vR#N!1N+p?M*0T}X7h{dXQc53#&j%kl4w0YtF~`Nk~^}3^`yPi437j2&=mjjhDvFhOC%uBKWHyvA)xjUaq$=N;4|1nT(<|wjena1{t*NHkc;)e;Qqb=*ZVhSI$)cw!7<%IE(iTzNbO0OwpGfBUgoHQ5(&|N`KC*~;joWxMuiyPgk%UI=3b`ZbkHKN-Sy;hb|(^?mNMoGojWZiiGSmm-Kc5ui~B~!G%}D#DiFOG>q*HLw_8!(5$f{jdB1<Kr`H*Iucu>ASImCB#Qga_95*v{${WxY15svV7)>`df}mwe(=x!Oo-)AW+S5WbPF{Zb+09JBo>H(|v^;L!Rrv+G?iXz?BX5$(<JQWFA#*-bd84_gYltO-!J>B-hfcz<&LXO_`_;87<wE}xgQAAY(@1!fLIe7ZC-XqVE<y<kEJM6}Bm-MRr`Y6Hi;>NlgXZsy@#3VLhx<4hKa!iz<_}XwJC}wpWUX;XeHXhCTdlwzBk*@DtojGk{%^Fj)!SQC01juY&+>dv#L?Q*r;Sw<R^rWe;hrI8u=a!G9u7|p_eG3WgyOSW-O&P*n1JFEIV>g!h?3;Ja_KUypIwH3SzAO=hA`<PhZSl({RE}XbS>M25kZMum20otBes`pZs!rR1c%2cjD>8<Hbb${Ozv39oTGqvY+|wiO7}Kh_U;=I14`|Rl=>A@=rQGPUL&w%%#u4n8-cIup6s&Xel?QHatS;}7?f9)P*dV>k!y&guN1&e)pB3bWd1B>!xHk2Os*aFPoXfz&SbIVhy=SbQm*7DCgA9ziM1_)o_}=&0v6YZFC(^ZK3ar_2t5esao-5hPNv(7%5V#yX|KRm(vHZEv~*KjgvMJ5`5#RD!<VbPMd`9pA+SvtY8>f*g}bD0e|lgPXyuW<cL&HWG=8iE_1NUWW(8rfdunTPGSVX>bXAV_43C<DRkbK+>~rJt7aW{Xvy<`#K?{$uQgS>rluUIZyZq1R=Y~3%b0%F6X?+uk{iG`qs?RuV!2PHJB*{Dw6s%blYSw^wXxnl=-M(}nl;b0&VlU-pfZ#$|4Er(wRTs4d?i0b?Slhh0wiG%}a2DG;6XOAW^Du}a`mquk2BjXiRJBE+YO|3+9#+93j%a!2Oljt8Orx*SZKhP{Gw<B3o;j^%^*DvG@a212e9gJm<=@Tb+b81F(gn1ZiOMTB)9+;u=oZ#A=)4L?4CigMvrP8}o>?WGOSy77STvvzSy1Ok@@^>lTW9smO6{D-f!S};l0QN&{p<x!>!1=ku$AxT#zyU<6;?g6{#wM&_ud1|WG4O)ki4zmNNtlatqH2bAjZyc`rSUj#qY~$^Gy0LJrVLK1Rg$5X18Bq-~Q_8Z>%%XJ>D7)TtBZ)YuEp792h(gB$EFni)Ccxq{01atM^N8B=Qmt4i9Jx_-Ba@YF%I5>PIaxT+rLCv1KNNUaXa6Zgr_OKgCvx2U9)19Iwr39vMAa{_qIgMe0`-AD>+>xae5eb&F(A#m<Dtm!S%Dm<77L1$0kE_zue<6jeh&yY=<-J3#b`ps>*vWz&*a<HXA2nn#|0owrvg{j$aA8yHOw4<8sBZE~rJE2>pPA$=Z#HC;Y~X^V!GT@$>n9aK*D>`0L_gQSW*K@QPdVs568pP7nah|&KaDCMkV'
_SEALED_BASE_SHA256 = '7de4b4f52c19d6e15f3d3927db5dc48c30f8556d2f2d151ebc95518ae125d340'


def _load_sealed_base():
    """解封可信的本项目历史定义；不联网、不安装、不执行旧main。"""
    content = zlib.decompress(base64.b85decode(_SEALED_BASE.encode('ascii')))
    if hashlib.sha256(content).hexdigest() != _SEALED_BASE_SHA256:
        raise RuntimeError('严格E1基座快照校验失败')
    exec(compile(content, '<sealed_strict_e1_definitions>', 'exec'), globals())


_load_sealed_base()
_base_install_classes = install_learning_classes
_base_episode_metrics = episode_metrics

FULL_VERSION = 'E01_FULL5_4SEEDS_120M_20261022_V1'
FULL_METHODS = ('SINGLE_EVENTGRAPH_FULL', 'SINGLE_EVENTQ_FULL',
                'J3_MATCHED_BASE', 'J3_TDM_EVENT_FULL', 'J3_ICM_AC_FULL')
FULL_SEEDS = (701, 702, 703, 704)
FULL_RULES = ('AUTHOR_SPF', 'AUTHOR_STTF', 'AUTHOR_QTTI', 'PURE_NEAREST',
              'READY_QUEUE_LQ', 'READY_QUEUE_JOINT')
FULL_ROOT_NAME = 'uam_E01_full5_4seeds_120m_v1'
FULL_ACTUAL_STEPS = 1_003_520
FULL_LATE = (850_000, 900_000, 950_000, FULL_ACTUAL_STEPS)
FULL_RUNTIME_PID = None


class FullSchema:
    """每个K都有显式字段边界；事件容量不依赖训练数据或未来请求。"""
    def __init__(self, k):
        self.k = int(k)
        self.frame = 4 + 8 * self.k
        self.history = 6 * self.frame
        self.passenger_slots, self.aircraft_slots = 300, 64
        self.event_slots = self.passenger_slots + self.aircraft_slots
        offset = self.history
        self.od = slice(offset, offset + 4); offset += 4
        self.resource_dim = 10 * self.k + 8
        self.resource = slice(offset, offset + self.resource_dim); offset += self.resource_dim
        self.tdm = slice(offset, offset + 5 * self.k); offset += 5 * self.k
        self.times = slice(offset, offset + self.k * self.event_slots); offset += self.k * self.event_slots
        self.masks = slice(offset, offset + self.k * self.event_slots); offset += self.k * self.event_slots
        self.passenger_active = offset
        self.pack = offset + 1
        self.return_mask = slice(2 * self.pack, 2 * self.pack + self.k)
        self.dim = 2 * self.pack + self.k

    def manifest(self):
        return dict(k=self.k, frame=self.frame, history=self.history, pack=self.pack, dim=self.dim,
                    resource_dim=self.resource_dim, passenger_slots=self.passenger_slots,
                    aircraft_slots=self.aircraft_slots,
                    contract='前半仅供乘客/critic；后半为原dispatch点状态，仅供飞机actor')


def full_task_registry():
    """以环境、K、seed为区组轮换方法，不按结果改顺序。"""
    tasks = []
    blocks = [(e, k, s) for e in ('E1', 'E0') for k in (3, 4, 2) for s in FULL_SEEDS]
    for block, (environment, k, seed) in enumerate(blocks):
        order = FULL_METHODS[block % 5:] + FULL_METHODS[:block % 5]
        for method in order:
            tasks.append(dict(id=f'{environment}__K{k}__{method}__s{seed}',
                              environment=environment, k=k, fleet=12 if environment == 'E1' else None,
                              method=method, seed=seed, block=block,
                              control='J3' if method.startswith('J3') else 'LQ'))
    assert len(tasks) == 120 and len({t['id'] for t in tasks}) == 120
    return tasks


def _source_step_generator(scenario):
    """只在作者补机/返航调用前插入yield，其余源step语句不动。"""
    owner = next(c for c in type(scenario).__mro__ if c.__name__ == 'Scenario')
    method = owner.step
    module = ast.parse(textwrap.dedent(inspect.getsource(method)))
    function = module.body[0]
    function.name = '_full_phased_source_step'
    hits = [i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == '_maintain_evtol_capacity']
    if len(hits) != 1:
        raise RuntimeError('作者step调度点不再唯一，拒绝猜测联合接口位置')
    function.body.insert(hits[0], ast.Expr(value=ast.Yield(value=ast.Constant(None))))
    ast.fix_missing_locations(module)
    namespace = dict(method.__globals__)
    exec(compile(module, '<source_step_with_one_yield>', 'exec'), namespace)
    return namespace[function.name]


def initialize_full_runtime():
    """只在worker中导入学习库；环境子进程不创建CUDA模型。"""
    global FULL_RUNTIME_PID, FullEnv, FullFeatures, FullPolicy, FullPPO, FullVecNormalize, FullCheckpoint
    if FULL_RUNTIME_PID == os.getpid():
        return
    _base_install_classes()
    from torch import nn
    import torch.nn.functional as F
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.distributions import CategoricalDistribution
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import obs_as_tensor
    from stable_baselines3.common import logger as sb_logger

    class FullEnv(gym.Env):
        """以generator暂停真实源分钟，不额外推进时钟、不复制奖励。"""
        metadata = {}

        def __init__(self, root, k, horizon, trace, method, environment, rank=0):
            self.k, self.schema = int(k), FullSchema(k)
            self.environment, self.method, self.rank = environment, method, rank
            self.joint = method.startswith('J3') and environment == 'E1'
            os.environ['UAM_STRICT_MODE'] = environment
            os.environ['UAM_STRICT_FLEET'] = '12' if environment == 'E1' else '0'
            self.legacy = SourceEnv(root, k, horizon, trace)
            self.raw, self.scenario = self.legacy.raw, self.legacy.scenario
            self.action_space = gym.spaces.Discrete(k * k)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.schema.dim,), np.float32)
            self._generator_function = _source_step_generator(self.scenario)
            self._generator = None
            self._old_dispatch = getattr(self.scenario, '_strict_dispatch_returns', None)
            if self._old_dispatch is not None:
                self.scenario._strict_dispatch_returns = MethodType(self._dispatch_hook, self.scenario)
            self.log_details = os.environ.get('UAM_STRICT_LOG_EVENTS') == '1'
            self.reset_diagnostics()

        def reset_diagnostics(self):
            self.control_stats = dict(dispatch_calls=0, effective_return_choices=0,
                                      priority_counts=[0] * self.k, actual_return_counts=[0] * self.k,
                                      event_overflow=0, peak_demand_events=0, peak_supply_events=0)
            self.control_trace, self.observed_od = [], []

        def _dispatch_hook(self, scenario):
            # 无学习控制或显式LQ验算时调用原严格E1的整段LQ函数。
            if not self.joint or getattr(self, '_priority', -1) < 0:
                return self._old_dispatch()
            from at_obj.evtol.vehicle_state import VehicleState
            candidates = tuple(str(v) for v in CANDIDATES[self.k])
            queue = {v: len(scenario.vertiports.vertiport_list[v].person_list) for v in candidates}
            available = sorted((e for e in scenario.vertiports.evtols_at_vertiport.get('2', [])
                                if e.state == VehicleState.IDLE and not e.passenger_ids), key=lambda e: str(e.id))
            requested = candidates[self._priority]
            self.control_stats['dispatch_calls'] += 1
            if len([v for v in candidates if queue[v] > 0]) > 1 and available:
                self.control_stats['effective_return_choices'] += 1
                self.control_stats['priority_counts'][self._priority] += 1
            for plane in available:
                ready = {v: sum(e.state == VehicleState.IDLE and not e.passenger_ids
                                for e in scenario.vertiports.evtols_at_vertiport[v]) for v in candidates}
                target = requested if queue[requested] > 0 else max(
                    candidates, key=lambda v: (queue[v], -ready[v], -int(v)))
                if queue[target] <= 0:
                    break
                if scenario._strict_start_empty(plane, target):
                    queue[target] = max(0, queue[target] - int(plane.spec.capacity))
                    self.control_stats['actual_return_counts'][candidates.index(target)] += 1
                    if self.log_details:
                        self.control_trace.append(dict(time=scenario.time, aircraft=str(plane.id),
                            priority=requested, executed=target, fallback=target != requested))

        def _events_and_pack(self, history, focal=None):
            s, d = self.scenario, self.schema
            if focal is None and s.waiting_decisions:
                focal = s.persons.persons[s.waiting_decisions[0]]
            od = list(focal.origin_position) + list(focal.destination_position) if focal else [0.] * 4
            history = np.asarray(history, dtype=np.float32).reshape(-1).copy()
            if focal is not None:
                history[-d.frame:-d.frame + 4] = od
            resources, tdm, times, masks = [], [], [], []
            for vid in CANDIDATES[self.k]:
                key = str(vid); station = s.vertiports.vertiport_list[key]
                incoming = [max(0., float(p.current_timer or 0.)) for p in s.persons.persons.values()
                            if str(getattr(p, 'origin_vertiport_id', '')) == key
                            and getattr(p, 'state', '') == 'enroute'
                            and getattr(p, 'sub_state', '') == 'to_vertiport']
                supply, idle, charging, flying = [], 0, 0, 0
                batteries = []
                for plane in s._all_evtols.values():
                    state = plane.state.name
                    here = str(plane.current_vertiport_id) == key
                    arriving = state == 'FLYING' and str(getattr(plane, 'target_vertiport_id', '')) == key
                    if plane.passenger_ids or not (here or arriving):
                        continue
                    if arriving:
                        from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
                        rate = float(VERTIPORT_CHARGE_RATE[key])
                        eta = math.ceil(max(0., float(plane.remaining_time)))
                        ready = eta + math.ceil(max(0., plane.spec.battery_capacity_kwh - plane.battery_kwh) / max(rate, 1e-8))
                        flying += 1
                    elif state == 'CHARGING':
                        ready = math.ceil(max(0., plane.spec.battery_capacity_kwh - plane.battery_kwh)
                                          / max(float(plane.spec.charge_rate_kwh_per_min), 1e-8))
                        charging += 1
                    elif state == 'IDLE':
                        ready = 0.; idle += 1
                    else:
                        continue
                    supply.append(float(ready)); batteries.append(float(plane.battery_kwh) / plane.spec.battery_capacity_kwh)
                incoming.sort(); supply.sort()
                self.control_stats['peak_demand_events'] = max(self.control_stats['peak_demand_events'], len(incoming))
                self.control_stats['peak_supply_events'] = max(self.control_stats['peak_supply_events'], len(supply))
                if len(incoming) > d.passenger_slots or len(supply) > d.aircraft_slots:
                    self.control_stats['event_overflow'] += 1
                    raise RuntimeError('已承诺事件超出声明容量；拒绝静默丢掉事件')
                times.extend([t / 60. for t in incoming] + [0.] * (d.passenger_slots - len(incoming))
                             + [t / 60. for t in supply] + [0.] * (d.aircraft_slots - len(supply)))
                masks.extend([1.] * len(incoming) + [0.] * (d.passenger_slots - len(incoming))
                             + [1.] * len(supply) + [0.] * (d.aircraft_slots - len(supply)))
                tau = float(s.vehicles.estimate_travel_time(focal.origin_position, station.vertiport_position)) if focal else 0.
                queue = float(len(station.person_list))
                demand = queue + sum(t <= tau for t in incoming) + float(focal is not None)
                available = float(sum(t <= tau for t in supply))
                tdm.extend([tau / 60., demand / 300., len(incoming) / 300., available / 20.,
                            (demand / (1. + available)) / 300.])
                resources.extend([queue / 300., len(incoming) / 300., idle / 20., charging / 20., flying / 20.,
                                  min(supply, default=600.) / 60., mean(supply) / 60. if supply else 10.,
                                  mean(batteries) if batteries else 0.,
                                  float(station.vertiport_position[0]) / 60., float(station.vertiport_position[1]) / 60.])
            destination_planes = s.vertiports.evtols_at_vertiport.get('2', [])
            resources.extend([sum(e.state.name == 'IDLE' and not e.passenger_ids for e in destination_planes) / 20.,
                              sum(e.state.name == 'CHARGING' for e in destination_planes) / 20.,
                              len(s._all_evtols) / 100., len(s.persons.persons) / 300.,
                              len(set(s.finished_ids)) / 300., float(s.time) / 600.,
                              float(self.environment == 'E1'), float(self.joint)])
            pack = np.concatenate([history, np.asarray(od + resources + tdm + times + masks
                                                      + [float(bool(s.waiting_decisions))], np.float32)])
            if pack.shape != (d.pack,) or not np.isfinite(pack).all():
                raise AssertionError('完整表示字段长度或有限性校验失败')
            return pack

        def _compose_pre(self):
            pack = self._events_and_pack(self.raw._get_stacked_obs())
            empty = np.zeros_like(pack)
            legal = np.zeros(self.k, np.float32); legal[0] = 1.
            return np.concatenate([pack, empty, legal])

        def reset(self, *, seed=None, options=None):
            if self._generator is not None:
                raise RuntimeError('不允许在暂停的分钟中reset')
            self.raw.reset(seed=seed, options=options)
            self.reset_diagnostics()
            self._episode_reward = 0.; self._episode_length = 0; self._episode_start = time.time()
            self.current = self._compose_pre()
            return self.current.copy(), {}

        def prepare_step(self, actions):
            if self._generator is not None:
                raise RuntimeError('同一分钟重复prepare')
            passenger_action = int(actions[self.rank] if isinstance(actions, (list, tuple, np.ndarray)) else actions)
            waiting = list(self.scenario.waiting_decisions)
            self._focal = self.scenario.persons.persons[waiting[0]] if waiting else None
            env_action = self.raw.decoder.decode(passenger_action, waiting[0]) if waiting else None
            if self._focal is not None:
                od = list(self._focal.origin_position) + list(self._focal.destination_position)
                if len(self.observed_od) < 64:
                    self.observed_od.append(od)
            s = self.scenario
            if hasattr(s, '_strict_initialized') and s._strict_initialized:
                s.strict_stats['charging_aircraft_minutes'] += sum(e.state.name == 'CHARGING' for e in s._all_evtols.values())
                s.strict_stats['idle_aircraft_minutes'] += sum(e.state.name == 'IDLE' for e in s._all_evtols.values())
            self._generator = self._generator_function(s, env_action)
            next(self._generator)
            # 只构造临时dispatch观测；正式history仍在finish时追加一次。
            frame = self.raw.encoder.encode(s, s.get_state())
            history = np.concatenate(list(self.raw.obs_buffer)[1:] + [frame])
            dispatch = self._events_and_pack(history, self._focal)
            legal = np.zeros(self.k, np.float32)
            available = [e for e in s.vertiports.evtols_at_vertiport.get('2', [])
                         if e.state.name == 'IDLE' and not e.passenger_ids]
            if self.joint and available:
                for i, vid in enumerate(CANDIDATES[self.k]):
                    legal[i] = float(bool(s.vertiports.vertiport_list[str(vid)].person_list))
            if not legal.any():
                legal[0] = 1.
            self._legal = legal
            self.pending = np.concatenate([self.current[:self.schema.pack], dispatch, legal])
            return self.pending.copy()

        def finish_step(self, actions):
            if self._generator is None:
                raise RuntimeError('finish之前没有prepare')
            priority = int(actions[self.rank] if isinstance(actions, (list, tuple, np.ndarray)) else actions)
            if self.joint and priority >= 0 and not self._legal[priority]:
                raise RuntimeError('飞机策略提交了mask之外的目标')
            self._priority = priority if self.joint else -1
            try:
                next(self._generator)
                raise AssertionError('源分钟出现多次dispatch暂停，拒绝重复计步')
            except StopIteration as done:
                state, reward, terminated, truncated, info = done.value
            finally:
                self._generator = None
            self.raw.state = state
            self.raw.obs_buffer.append(self.raw.encoder.encode(self.scenario, state))
            s = self.scenario
            if hasattr(s, '_strict_assert'):
                s._strict_assert()
                if s._strict_log and s.time % 5 == 0:
                    row = dict(time=s.time)
                    for vid in tuple(str(v) for v in CANDIDATES[self.k]) + ('2',):
                        local = s.vertiports.evtols_at_vertiport[vid]
                        row['V' + vid + '_queue'] = len(s.vertiports.vertiport_list[vid].person_list)
                        row['V' + vid + '_ready'] = sum(e.state.name == 'IDLE' for e in local)
                        row['V' + vid + '_charging'] = sum(e.state.name == 'CHARGING' for e in local)
                    s.strict_resources.append(row)
            self._episode_reward += float(reward); self._episode_length += 1
            self.current = self._compose_pre()
            info = dict(info)
            info['TimeLimit.truncated'] = bool(truncated and not terminated)
            if terminated or truncated:
                info['episode'] = dict(r=self._episode_reward, l=self._episode_length,
                                       t=time.time() - self._episode_start)
                info['terminal_observation'] = self.current.copy()
            return self.current.copy(), float(reward), bool(terminated or truncated), info

        def reset_selected(self, flags):
            return self.reset()[0] if flags[self.rank] else self.current.copy()

        def step(self, action):
            # 标准Gym接口仅用于外部检查；正式PPO与回放使用显式两阶段。
            self.prepare_step(int(action) // self.k)
            obs, reward, done, info = self.finish_step(int(action) % self.k)
            return obs, reward, False, done, info

        def counterfactual_pre(self, od):
            """用已经揭示过的OD重算tau/TDM；不只改一个不一致的OD字段。"""
            focal = SimpleNamespace(origin_position=od[:2], destination_position=od[2:])
            pack = self._events_and_pack(self.raw._get_stacked_obs(), focal)
            return np.concatenate([pack, np.zeros_like(pack), np.eye(self.k, dtype=np.float32)[0]])

        def close(self):
            if self._generator is not None:
                self._generator.close(); self._generator = None
            self.raw.close()

    class FullVecNormalize(VecNormalize):
        """源history与OD归一化；手工缩放的事件/资源及mask保持物理单位。"""
        def normalize_obs(self, obs):
            value = super().normalize_obs(obs)
            schema = FullSchema(int(self.venv.get_attr('k', indices=[0])[0])) if not hasattr(self, '_full_schema') else self._full_schema
            self._full_schema = schema
            for start in (0, schema.pack):
                lo = start + schema.resource.start
                value[..., lo:start + schema.pack] = np.asarray(obs)[..., lo:start + schema.pack]
            value[..., schema.return_mask] = np.asarray(obs)[..., schema.return_mask]
            return value.astype(np.float32)

    class FullFeatures(BaseFeaturesExtractor):
        """公共原始输入相同；完整机制支路按方法激活，不借未来dispatch影响乘客。"""
        def __init__(self, observation_space, k=2, method='J3_MATCHED_BASE', environment='E0'):
            super().__init__(observation_space, 128)
            from utilss.encoding import TemporalLSTMExtractor
            self.k, self.method, self.environment = k, method, environment
            self.schema = d = FullSchema(k)
            self.temporal = TemporalLSTMExtractor(gym.spaces.Box(-np.inf, np.inf, (6, d.frame), np.float32), features_dim=128)
            self.resource_net = nn.Sequential(nn.Linear(d.resource_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.focal_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.core = nn.Sequential(nn.Linear(224, 192), nn.ReLU(), nn.Linear(192, 128), nn.ReLU())
            raw_dim = 2 * k * d.event_slots + 5 * k
            self.raw_events = nn.Sequential(nn.Linear(raw_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.common_fuse = nn.Sequential(nn.Linear(192, 128), nn.ReLU())
            self.final_fuse = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Linear(192, 128), nn.ReLU())
            self.event_phi = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.query = nn.Sequential(nn.Linear(1 + k, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.event_out = nn.Sequential(nn.Linear(k * 32, 64), nn.ReLU())
            if method == 'SINGLE_EVENTGRAPH_FULL':
                self.graph_node = nn.Sequential(nn.Linear(37, 64), nn.ReLU())
                self.graph_attention = nn.MultiheadAttention(64, 4, batch_first=True)
                self.graph_out = nn.Sequential(nn.Linear(k * 64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
            if method == 'J3_TDM_EVENT_FULL':
                self.tdm_net = nn.Sequential(nn.Linear(5 * k, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
                self.tdm_event_fuse = nn.Sequential(nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
            if method == 'J3_ICM_AC_FULL':
                self.icm_state = nn.Sequential(nn.Linear(d.resource_dim + 5 * k, 96), nn.ReLU(), nn.Linear(96, 32), nn.ReLU())
                self.icm_inverse = nn.Sequential(nn.Linear(64, 96), nn.ReLU(), nn.Linear(96, k * k))
                self.icm_action = nn.Sequential(nn.Linear(32 + k * k, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
                self.icm_readout = nn.Sequential(nn.Linear(k * k * 16, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
            self.register_buffer('event_types', torch.cat([torch.zeros(d.passenger_slots), torch.ones(d.aircraft_slots)]))

        def auxiliary_parameters(self):
            return list(self.icm_state.parameters()) + list(self.icm_inverse.parameters()) if self.method == 'J3_ICM_AC_FULL' else []

        def icm_z(self, pack):
            d = self.schema
            return self.icm_state(torch.cat([pack[:, d.resource], pack[:, d.tdm]], -1))

        def encode_pack(self, pack):
            d = self.schema; batch = pack.shape[0]
            temporal = self.temporal(pack[:, :d.history].reshape(batch, 6, d.frame))
            core = self.core(torch.cat([temporal, self.resource_net(pack[:, d.resource]), self.focal_net(pack[:, d.od])], -1))
            times = pack[:, d.times].reshape(batch, self.k, d.event_slots)
            mask = pack[:, d.masks].reshape(batch, self.k, d.event_slots)
            tdm = pack[:, d.tdm].reshape(batch, self.k, 5)
            raw = torch.cat([(times * mask).flatten(1), mask.flatten(1), tdm.flatten(1)], -1)
            common = self.common_fuse(torch.cat([core, self.raw_events(raw)], -1))
            mechanism = torch.zeros((batch, 64), dtype=pack.dtype, device=pack.device)
            if self.method in ('SINGLE_EVENTGRAPH_FULL', 'SINGLE_EVENTQ_FULL', 'J3_TDM_EVENT_FULL'):
                types = self.event_types.view(1, 1, -1).expand_as(times)
                embeddings = self.event_phi(torch.stack([times, types], -1))
                if self.method == 'SINGLE_EVENTGRAPH_FULL':
                    station = (embeddings * mask[..., None]).sum(2) / mask.sum(2).clamp_min(1.)[..., None]
                    nodes = self.graph_node(torch.cat([station, tdm], -1))
                    nodes, _ = self.graph_attention(nodes, nodes, nodes, need_weights=False)
                    mechanism = self.graph_out(nodes.flatten(1))
                else:
                    ids = torch.eye(self.k, device=pack.device).expand(batch, -1, -1)
                    query = self.query(torch.cat([tdm[..., :1], ids], -1))
                    logits = (embeddings * query[:, :, None, :]).sum(-1) / math.sqrt(32.)
                    weights = torch.softmax(logits.masked_fill(mask < .5, -1e9), -1) * mask
                    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
                    event = self.event_out((embeddings * weights[..., None]).sum(2).flatten(1))
                    mechanism = self.tdm_event_fuse(torch.cat([self.tdm_net(tdm.flatten(1)), event], -1)) if self.method == 'J3_TDM_EVENT_FULL' else event
            elif self.method == 'J3_ICM_AC_FULL':
                z = self.icm_z(pack); actions = torch.eye(self.k * self.k, device=pack.device)
                candidate = self.icm_action(torch.cat([z[:, None, :].expand(-1, self.k * self.k, -1),
                                                       actions[None].expand(batch, -1, -1)], -1))
                if self.environment == 'E0':
                    keep = (torch.arange(self.k * self.k, device=pack.device) % self.k == 0).float()
                    candidate = candidate * keep[None, :, None]
                mechanism = self.icm_readout(candidate.flatten(1))
            return self.final_fuse(torch.cat([common, mechanism], -1))

        def forward(self, observations):
            return self.encode_pack(observations[:, :self.schema.pack].float())

        def dispatch_features(self, observations):
            d = self.schema
            return self.encode_pack(observations[:, d.pack:2 * d.pack].float())

    class FullPolicy(ActorCriticPolicy):
        """乘客/critic只读前态；飞机actor读真实dispatch态并条件于乘客动作。"""
        def _build(self, lr_schedule):
            super()._build(lr_schedule)
            self.k = self.features_extractor.k
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.k)
            self.return_head = nn.Sequential(nn.Linear(128 + self.k, 256), nn.Tanh(),
                                             nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, self.k))
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        def passenger_parts(self, obs):
            pre = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(pre)
            logits = self.action_net(latent_pi)
            active = obs[:, self.features_extractor.schema.passenger_active] > .5
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[~active] = False; mask[~active, 0] = True
            logits = logits.masked_fill(~mask, -1e9)
            return logits, self.value_net(latent_vf)

        def sample_passenger(self, obs, deterministic=False):
            logits, values = self.passenger_parts(obs)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = logits.argmax(-1) if deterministic else distribution.sample()
            return actions, values, distribution.probs

        def joint_logits(self, obs):
            passenger, _ = self.passenger_parts(obs)
            d = self.features_extractor.schema; batch = obs.shape[0]
            dispatch = self.features_extractor.dispatch_features(obs)
            ids = torch.eye(self.k, device=obs.device)[None].expand(batch, -1, -1)
            return_logits = self.return_head(torch.cat([dispatch[:, None, :].expand(-1, self.k, -1), ids], -1))
            legal = obs[:, d.return_mask] > .5
            return_logits = return_logits.masked_fill(~legal[:, None, :], -1e9)
            return (F.log_softmax(passenger, -1)[:, :, None] + F.log_softmax(return_logits, -1)).flatten(1)

        def sample_return(self, obs, passenger, deterministic=False):
            returning, _ = self.return_with_logprob(obs, passenger, deterministic)
            return returning

        def return_with_logprob(self, obs, passenger, deterministic=False):
            # 采样只计算已选乘客动作的飞机头，避免一tick重复跑五次表示网络。
            distribution = self.return_distribution(obs, passenger)
            returning = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            return returning, distribution.log_prob(returning)

        def return_distribution(self, obs, passenger):
            dispatch = self.features_extractor.dispatch_features(obs)
            ids = F.one_hot(passenger.long(), self.k).to(dispatch.dtype)
            chosen = self.return_head(torch.cat([dispatch, ids], -1))
            d = self.features_extractor.schema
            chosen = chosen.masked_fill(obs[:, d.return_mask] < .5, -1e9)
            return torch.distributions.Categorical(logits=chosen)

        def get_distribution(self, obs):
            return CategoricalDistribution(self.k * self.k).proba_distribution(action_logits=self.joint_logits(obs))

        def forward(self, obs, deterministic=False):
            distribution = self.get_distribution(obs)
            actions = distribution.get_actions(deterministic=deterministic)
            return actions, self.predict_values(obs), distribution.log_prob(actions)

        def evaluate_actions(self, obs, actions):
            # dispatch态只对应实际乘客动作；不能用它臆造其他乘客动作的后继态熵。
            actions = actions.long().flatten()
            passenger, returning = actions // self.k, actions % self.k
            logits, values = self.passenger_parts(obs)
            p_dist = torch.distributions.Categorical(logits=logits)
            r_dist = self.return_distribution(obs, passenger)
            return values, p_dist.log_prob(passenger) + r_dist.log_prob(returning), p_dist.entropy() + r_dist.entropy()

        def predict_values(self, obs):
            _, values = self.passenger_parts(obs)
            return values

    class FullPPO(PPO):
        """两个零时长子决策合成一个物理transition，沿用PPO原目标。"""
        def _setup_model(self):
            super()._setup_model()
            params = self.policy.features_extractor.auxiliary_parameters()
            self.aux_optimizer = torch.optim.Adam(params, lr=1e-3) if params else None
            self._full_save_hook = None

        def _excluded_save_params(self):
            return super()._excluded_save_params() + ['_full_save_hook', 'aux_optimizer', '_last_raw_full']

        def _get_torch_save_params(self):
            states, variables = super()._get_torch_save_params()
            if getattr(self, 'aux_optimizer', None) is not None:
                states += ['aux_optimizer']
            return states, variables

        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
            if self.use_sde:
                raise RuntimeError('本矩阵为离散动作，不支持额外SDE协议')
            self.policy.set_training_mode(False)
            rollout_buffer.reset(); callback.on_rollout_start()
            vector = env.venv
            raw = getattr(self, '_last_raw_full', env.get_original_obs())
            for n_steps in range(n_rollout_steps):
                pre_norm = env.normalize_obs(raw.copy())
                with torch.no_grad():
                    passenger, values, passenger_probs = self.policy.sample_passenger(obs_as_tensor(pre_norm, self.device))
                pending_raw = np.stack(vector.env_method('prepare_step', passenger.cpu().numpy().tolist()))
                if env.training and env.norm_obs:
                    env.obs_rms.update(pending_raw)
                pending = env.normalize_obs(pending_raw.copy())
                d = self.policy.features_extractor.schema
                pending[:, :d.pack] = pre_norm[:, :d.pack]
                tensor = obs_as_tensor(pending, self.device)
                with torch.no_grad():
                    returning, return_logprob = self.policy.return_with_logprob(tensor, passenger)
                    actions = passenger * self.policy.k + returning
                    passenger_logprob = passenger_probs.gather(1, passenger[:, None]).flatten().clamp_min(1e-30).log()
                    log_probs = passenger_logprob + return_logprob
                result = vector.env_method('finish_step', returning.cpu().numpy().tolist())
                new_raw = np.stack([r[0] for r in result]); raw_reward = np.asarray([r[1] for r in result], np.float32)
                dones = np.asarray([r[2] for r in result], bool); infos = [r[3] for r in result]
                env.returns = env.returns * env.gamma + raw_reward
                if env.training and env.norm_reward:
                    env.ret_rms.update(env.returns)
                rewards = env.normalize_reward(raw_reward.copy())
                env.returns[dones] = 0.
                env.old_obs = new_raw.copy(); env.old_reward = raw_reward.copy()
                new_obs = env.normalize_obs(new_raw.copy())
                for i, done in enumerate(dones):
                    if done and infos[i].get('TimeLimit.truncated'):
                        terminal = env.normalize_obs(infos[i]['terminal_observation'][None])[0]
                        with torch.no_grad():
                            terminal_value = self.policy.predict_values(obs_as_tensor(terminal[None], self.device))[0]
                        rewards[i] += self.gamma * float(terminal_value)
                self.num_timesteps += env.num_envs
                self._update_info_buffer(infos, dones)
                rollout_buffer.add(pending, actions.cpu().numpy().reshape(-1, 1), rewards,
                                   self._last_episode_starts, values, log_probs)
                if dones.any():
                    new_raw = np.stack(vector.env_method('reset_selected', dones.tolist()))
                    new_obs = env.normalize_obs(new_raw.copy())
                self._last_obs, self._last_raw_full = new_obs, new_raw
                self._last_episode_starts = dones
                self._last_original_obs = new_raw
                raw = new_raw
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
            with torch.no_grad():
                last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
            rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
            callback.update_locals(locals()); callback.on_rollout_end()
            return True

        def train(self):
            super().train()
            if self.aux_optimizer is not None:
                b = self.rollout_buffer
                obs = b.observations
                acts = b.actions
                if b.generator_ready:
                    obs = obs.reshape(b.n_envs, b.buffer_size, -1).swapaxes(0, 1)
                    acts = acts.reshape(b.n_envs, b.buffer_size, -1).swapaxes(0, 1)
                else:
                    obs = obs.reshape(b.buffer_size, b.n_envs, -1)
                    acts = acts.reshape(b.buffer_size, b.n_envs, -1)
                valid = np.argwhere(b.episode_starts[1:] < .5)
                if len(valid) > 12000:
                    valid = valid[np.random.choice(len(valid), 12000, replace=False)]
                np.random.shuffle(valid); losses = []
                ext = self.policy.features_extractor; d = ext.schema
                for start in range(0, len(valid), 2048):
                    idx = valid[start:start + 2048]
                    cur = obs_as_tensor(obs[idx[:, 0], idx[:, 1]], self.device)
                    nxt = obs_as_tensor(obs[idx[:, 0] + 1, idx[:, 1]], self.device)
                    action = torch.as_tensor(acts[idx[:, 0], idx[:, 1], 0], device=self.device).long()
                    logits = ext.icm_inverse(torch.cat([ext.icm_z(cur[:, :d.pack]), ext.icm_z(nxt[:, :d.pack])], -1))
                    log_joint = F.log_softmax(logits, -1).reshape(-1, ext.k, ext.k)
                    p_active = cur[:, d.passenger_active] > .5
                    r_active = (cur[:, d.return_mask] > .5).sum(-1) > 1
                    a_p, a_r = action // ext.k, action % ext.k
                    selected = log_joint[torch.arange(len(idx), device=self.device), a_p, a_r]
                    p_marginal = torch.logsumexp(log_joint, -1).gather(1, a_p[:, None]).flatten()
                    r_marginal = torch.logsumexp(log_joint, -2).gather(1, a_r[:, None]).flatten()
                    likelihood = torch.where(p_active & r_active, selected,
                                              torch.where(p_active, p_marginal, r_marginal))
                    active = p_active | r_active
                    if not active.any():
                        continue
                    loss = -likelihood[active].mean()
                    self.aux_optimizer.zero_grad(set_to_none=True)
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ext.auxiliary_parameters(), 5.)
                    self.aux_optimizer.step()
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    losses.append(float(loss.detach()))
                self.logger.record('train/icm_inverse_loss', mean(losses) if losses else 0.)
                self.logger.record('train/icm_aux_samples', len(valid))
            if self._full_save_hook is not None:
                self._full_save_hook(self)

    class FullCheckpoint(BaseCallback):
        def __init__(self, directory):
            super().__init__(); self.directory = Path(directory)
        def _on_training_start(self):
            self.next = (self.model.num_timesteps // 50000 + 1) * 50000
        def _on_step(self):
            if self.model.num_timesteps >= self.next:
                full_save_pair(self.model, self.directory / 'checkpoints', self.model.num_timesteps)
                self.next += 50000
            return True

    for cls in (FullEnv, FullVecNormalize, FullFeatures, FullPolicy, FullPPO, FullCheckpoint):
        cls.__qualname__ = cls.__name__; cls.__module__ = __name__
    FULL_RUNTIME_PID = os.getpid()


class FullFactory:
    """spawn可恢复工厂，只在本cell环境内设置物理开关。"""
    def __init__(self, root, task, config, rank):
        self.root, self.task, self.config, self.rank = str(root), task, config, rank
    def __call__(self):
        initialize_full_runtime()
        from stable_baselines3.common.utils import set_random_seed
        t, c = self.task, self.config
        os.environ['UAM_STRICT_MODE'] = t['environment']
        os.environ['UAM_STRICT_FLEET'] = '12' if t['environment'] == 'E1' else '0'
        set_random_seed(t['seed'] + self.rank)
        from loguru import logger
        logger.remove(); logging.disable(logging.CRITICAL)
        return FullEnv(self.root, t['k'], c['train_horizon'], c['trace'], t['method'], t['environment'], self.rank)


def full_save_pair(model, directory, step):
    """分析快照和归一化使用同名原子指针；文件不属于数据压缩包。"""
    pair = save_pair(model, directory, step)
    return pair


def full_resume_hook(directory):
    def save(model):
        pair = full_save_pair(model, Path(directory) / 'resume', model.num_timesteps)
        rng_file = Path(directory) / 'resume' / ('rng_' + str(model.num_timesteps) + '.pt')
        state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else None)
        torch.save(state, rng_file)
        pair['rng'] = str(rng_file)
        pair['continuation_note'] = '完整rollout恢复；重启环境episode，不声称逐bit继续'
        save_json(Path(directory) / 'RESUME_POINTER.json', pair)
        save_json(Path(directory) / 'LIVE.json', dict(phase='train', steps=model.num_timesteps,
                  time=now(), pid=os.getpid()))
        # 分析检查点全部保留；只保留最近两次完整rollout恢复件，控制磁盘占用。
        resume_dir = (Path(directory) / 'resume').resolve()
        markers = sorted(resume_dir.glob('step_*.json'))
        for marker in markers[:-2]:
            record = read_json(marker)
            files = [record['model'], record['vecnorm'], str(marker),
                     str(resume_dir / ('rng_' + str(record['step']) + '.pt'))]
            for name in files:
                raw_path = Path(name)
                owned = (raw_path if raw_path.is_absolute() else resume_dir / raw_path).resolve()
                if owned.parent != resume_dir:
                    raise RuntimeError('恢复件路径离开当前run目录，拒绝清理')
                if owned.is_file():
                    owned.unlink()
    return save


def full_setup(task, root):
    directory = Path(root) / 'cells' / task['id']; directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    os.environ['UAM_STRICT_MODE'] = task['environment']
    os.environ['UAM_STRICT_FLEET'] = '12' if task['environment'] == 'E1' else '0'
    initialize_full_runtime(); activate_source(root, task['k'])
    from loguru import logger
    logger.remove(); logging.disable(logging.CRITICAL)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    return directory


def full_train(task, root, config):
    directory = full_setup(task, root)
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.logger import configure
    set_random_seed(task['seed'])
    if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('请求CUDA但不可用；不会静默切CPU')
    factories = [FullFactory(root, task, config, rank) for rank in range(5)]
    vector = SubprocVecEnv(factories, start_method='spawn')
    norm = None; started = time.time()
    try:
        pointer = read_json(directory / 'RESUME_POINTER.json') if config['resume'] else None
        if pointer:
            norm = FullVecNormalize.load(pointer['vecnorm'], vector)
            norm.training = True; norm.norm_reward = True
            model = FullPPO.load(pointer['model'], env=norm, device=config['device'], force_reset=True)
            if model.num_timesteps != pointer['step']:
                raise RuntimeError('恢复模型与归一化步数不匹配')
            rng = torch.load(pointer['rng'], map_location='cpu', weights_only=False)
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None and str(model.device).startswith('cuda'):
                torch.cuda.set_rng_state_all(rng['cuda'])
            old = read_json(directory / 'RECOVERY.json', dict(restarts=0))
            old.update(restarts=old['restarts'] + 1, resumed_at=model.num_timesteps, time=now())
            save_json(directory / 'RECOVERY.json', old)
        else:
            norm = FullVecNormalize(vector, norm_obs=True, norm_reward=True, gamma=.99, clip_obs=10., clip_reward=10.)
            kwargs = {k: v for k, v in SOURCE_PPO.items() if k not in ('n_envs', 'learning_rate')}
            model = FullPPO(FullPolicy, norm, learning_rate=lambda progress:5e-4 * max(0., progress),
                policy_kwargs=dict(features_extractor_class=FullFeatures,
                    features_extractor_kwargs=dict(k=task['k'], method=task['method'], environment=task['environment']),
                    net_arch=dict(pi=[256, 256], vf=[256, 256])), seed=task['seed'], device=config['device'], verbose=1, **kwargs)
        if model.policy.features_extractor.schema.dim != norm.observation_space.shape[0]:
            raise AssertionError('schema与VecNormalize维度不一致')
        segment = directory / 'training' / ('segment_' + str(time.time_ns()))
        model.set_logger(configure(str(segment), ['stdout', 'csv', 'tensorboard']))
        model._full_save_hook = full_resume_hook(directory)
        save_json(directory / 'MODEL_MANIFEST.json', dict(task=task, config=config,
            parameters=sum(p.numel() for p in model.policy.parameters()), schema=FullSchema(task['k']).manifest(),
            original_reward=True, aircraft_branch_active=task['control'] == 'J3' and task['environment'] == 'E1',
            auxiliary='inverse+action-conditioned-readout; no intrinsic reward', source_commit=source_snapshot()['commit'],
            ppo=SOURCE_PPO, code_sha256=digest(SCRIPT), base_sha256=_SEALED_BASE_SHA256))
        remaining = FULL_ACTUAL_STEPS - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=FullCheckpoint(directory), reset_num_timesteps=False)
        pair = full_save_pair(model, directory / 'checkpoints', model.num_timesteps)
        model._full_save_hook(model)
        save_json(directory / 'TRAIN_END.json', dict(task=task, steps=model.num_timesteps, final=pair,
                  wall_seconds=time.time() - started, status='TRAINED', time=now()))
    finally:
        (norm if norm is not None else vector).close()


def _full_rule_passenger(env, rule):
    s = env.scenario; candidates = CANDIDATES[env.k]
    if not s.waiting_decisions:
        return 0
    if rule.startswith('AUTHOR_'):
        if not hasattr(env, '_author_cache'):
            env._author_cache = {}
        if rule not in env._author_cache:
            env._author_cache[rule] = AuthorFunctions(rule.split('_', 1)[1])
        author = env._author_cache[rule]
        return adapter_decision(s, env.k, author, np)[0]
    person = s.persons.persons[s.waiting_decisions[0]]
    scores = []
    for vid in candidates:
        v = s.vertiports.vertiport_list[str(vid)]
        access = float(s.vehicles.estimate_travel_time(person.origin_position, v.vertiport_position))
        if rule == 'PURE_NEAREST':
            scores.append(abs(float(person.origin_position[0]) - float(v.vertiport_position[0]))
                          + abs(float(person.origin_position[1]) - float(v.vertiport_position[1])))
            continue
        distance = s.vertiports._calculate_distance(v.vertiport_position, s.vertiports.vertiport_list['2'].vertiport_position)
        flight = distance / 200. * 60.
        local = s.vertiports.evtols_at_vertiport.get(str(vid), [])
        ready = [max(0., e.spec.battery_capacity_kwh - e.battery_kwh) / max(e.spec.charge_rate_kwh_per_min, 1e-8)
                 for e in local if not e.passenger_ids and e.state.name in ('IDLE', 'CHARGING')]
        queued = len(v.person_list)
        incoming = sum(str(getattr(p, 'origin_vertiport_id', '')) == str(vid)
                       and getattr(p, 'sub_state', '') == 'to_vertiport'
                       and getattr(p, 'state', '') == 'enroute'
                       and max(0., float(p.current_timer or 0.)) <= access for p in s.persons.persons.values())
        # 这是在线启发式估计，不宣称知道未决返航的真实未来。
        cycle = 2. * flight + 2. * distance * .55 / 5.32
        service_wait = min(ready, default=flight + distance * .55 / 5.32)
        service_wait += ((queued + incoming) // 4) * cycle / max(1, len(ready))
        scores.append(max(access, service_wait) + flight)
    return min(range(env.k), key=lambda i:(scores[i], i))


def full_replay(model, norm, root, task, config, seed, deterministic=True, rule=None, probe=False):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    method = ('J3_MATCHED_BASE' if rule == 'READY_QUEUE_JOINT' else 'SINGLE_EVENTQ_FULL') if rule else task['method']
    capture_lifecycle = bool(rule or probe or config.get('_capture_lifecycle', False))
    os.environ['UAM_STRICT_LOG_EVENTS'] = '1' if capture_lifecycle else '0'
    env = FullEnv(root, task['k'], config['eval_horizon'], config['trace'], method, task['environment'])
    obs, _ = env.reset(seed=seed)
    counts = [0] * task['k']; area = 0.; probs = []; probe_tv = []; probe_flip = []
    try:
        while True:
            active = bool(env.scenario.waiting_decisions)
            area += sum(pid not in set(env.scenario.finished_ids) for pid in env.scenario.persons.persons)
            pre = norm.normalize_obs(obs[None]) if norm is not None else obs[None]
            if rule:
                passenger = _full_rule_passenger(env, rule)
            else:
                with torch.no_grad():
                    action, _, distribution = model.policy.sample_passenger(torch.as_tensor(pre, device=model.device), deterministic)
                    passenger = int(action[0]); probability = distribution[0].cpu().numpy()
                if active:
                    probs.append(probability.tolist())
                    if probe and env.observed_od and len(probe_tv) < 128:
                        counter = env.counterfactual_pre(env.observed_od[len(probe_tv) % len(env.observed_od)])
                        with torch.no_grad():
                            _, _, alternate = model.policy.sample_passenger(torch.as_tensor(norm.normalize_obs(counter[None]), device=model.device), True)
                        alternate = alternate[0].cpu().numpy()
                        probe_tv.append(float(np.abs(probability - alternate).sum() / 2.))
                        probe_flip.append(int(probability.argmax() != alternate.argmax()))
            if active:
                counts[passenger] += 1
            pending_raw = env.prepare_step(passenger)
            if rule:
                if rule == 'READY_QUEUE_JOINT' and env.joint:
                    valid = [i for i in range(env.k) if pending_raw[env.schema.return_mask][i] > .5]
                    returning = max(valid, key=lambda i:len(env.scenario.vertiports.vertiport_list[str(CANDIDATES[env.k][i])].person_list))
                else:
                    returning = -1
            else:
                pending = norm.normalize_obs(pending_raw[None])
                pending[:, :env.schema.pack] = pre[:, :env.schema.pack]
                with torch.no_grad():
                    returning = int(model.policy.sample_return(torch.as_tensor(pending, device=model.device),
                                    torch.as_tensor([passenger], device=model.device), deterministic)[0])
            obs, reward, done, info = env.finish_step(returning)
            if done:
                break
        result = _base_episode_metrics(env, counts, area)
        if not (rule or config.get('_final_checkpoint', False)):
            # 中期仅留资源摘要；完整逐飞机事件集中放final，减小数据包。
            result['aircraft_events'] = []
        result.update(eval_seed=seed, mode=rule or ('DET' if deterministic else 'STO'),
                      control_diagnostics=env.control_stats, control_trace=env.control_trace,
                      final_station_queues={str(v):len(env.scenario.vertiports.vertiport_list[str(v)].person_list) for v in CANDIDATES[env.k]},
                      final_queue_mod4={str(v):len(env.scenario.vertiports.vertiport_list[str(v)].person_list) % 4 for v in CANDIDATES[env.k]},
                      OD_legal_history_TV=mean(probe_tv) if probe_tv else None,
                      OD_legal_history_flip=mean(probe_flip) if probe_flip else None)
        if probs:
            ps = np.asarray(probs); mean_p = ps.mean(0)
            entropy = float(-(ps * np.log(np.maximum(ps, 1e-12))).sum(1).mean())
            result.update(mean_policy_p=mean_p.tolist(), entropy=entropy,
                          state_MI=float(-(mean_p * np.log(np.maximum(mean_p, 1e-12))).sum() - entropy))
        return result
    finally:
        env.close()


def _full_summary_step(step, episodes):
    row = dict(step=step)
    for mode in ('DET', 'STO'):
        group = [e for e in episodes if e['mode'] == mode]
        for name, key in (('co', 'ATT_completed_only'), ('completion', 'completion'),
                          ('rpt', 'restricted_person_time_per_generated'), ('wait', 'AWT_completed_only'),
                          ('ground', 'AGT_completed_only'), ('flight', 'AFT_completed_only')):
            row[mode + '_' + name] = mean([e[key] for e in group]) if group else None
        row[mode + '_ATT'] = mean([e['ATT'] for e in group]) if group and all(e['full'] for e in group) else None
    return row


def full_eval(task, root, config):
    directory = full_setup(task, root)
    import pickle
    checkpoints = [read_json(p) for p in sorted((directory / 'checkpoints').glob('step_*.json'))]
    if not checkpoints:
        raise RuntimeError('没有成对的分析检查点')
    curves = []; started = time.time()
    for cp in checkpoints:
        step = cp['step']; destination = directory / 'evaluation' / f'step_{step:09d}.json'
        existing = read_json(destination) if config['resume'] else None
        if existing:
            curves.append(existing['summary']); continue
        model = FullPPO.load(directory / 'checkpoints' / Path(cp['model']).name, device=config['eval_device'])
        # 只反序列化本实验自己生成、与模型同名配对的归一化文件。
        with (directory / 'checkpoints' / Path(cp['vecnorm']).name).open('rb') as f:
            norm = pickle.load(f)
        norm.training = False; norm.norm_reward = False
        if norm.observation_space.shape != (FullSchema(task['k']).dim,):
            raise RuntimeError('拒绝错K或错schema归一化文件')
        episodes = []
        replay_config = dict(config, _capture_lifecycle=step in FULL_LATE,
                             _final_checkpoint=step == FULL_ACTUAL_STEPS)
        for seed in (123, 124):
            episodes.append(full_replay(model, norm, root, task, replay_config, seed, True, probe=step == FULL_ACTUAL_STEPS))
            if step % 100000 == 0 or step in FULL_LATE:
                episodes.append(full_replay(model, norm, root, task, replay_config, seed, False))
        row = _full_summary_step(step, episodes); curves.append(row)
        save_json(destination, dict(summary=row, episodes=episodes, time=now()))
        save_csv(directory / 'curve.csv', curves)
        print(f"[回放] {task['id']} {step:,} DET-CO={row['DET_co']:.3f} 完成率={row['DET_completion']:.3%}", flush=True)
    late = [r for r in curves if r['step'] in FULL_LATE]
    if len(late) != len(FULL_LATE):
        raise RuntimeError('正式late四点不完整，不用近似点静默代替')
    result = dict(task=task, late_steps=list(FULL_LATE), time=now(), eval_seconds=time.time() - started)
    for mode in ('DET', 'STO'):
        values = [r[mode + '_co'] for r in late]
        all_values = [r[mode + '_co'] for r in curves if r[mode + '_co'] is not None]
        result[mode] = dict(late_co=mean(values), late_sd=sd(values), final_co=curves[-1][mode + '_co'],
            late_slope_per_million=strict_late_slope(late, mode + '_co'), best_co=min(all_values),
            best_to_late=mean(values) - min(all_values),
            completion=mean([r[mode + '_completion'] for r in late]),
            rpt=mean([r[mode + '_rpt'] for r in late]),
            all_ATT=mean([r[mode + '_ATT'] for r in late]) if all(r[mode + '_ATT'] is not None for r in late) else None)
    save_json(directory / 'CELL_SUMMARY.json', result)
    save_json(directory / 'EVAL_END.json', dict(time=now(), status='EVALUATED', task=task))


def full_rules(task, root, config):
    directory = full_setup(task, root)
    rows = []
    for rule in FULL_RULES:
        if task['environment'] == 'E0' and rule == 'READY_QUEUE_JOINT':
            continue
        for seed in (123, 124):
            path = directory / 'episodes' / f'{rule}__s{seed}.json'
            row = read_json(path) if config['resume'] else None
            if row is None:
                row = full_replay(None, None, root, task, config, seed, rule=rule)
                save_json(path, row)
            rows.append(row)
    save_json(directory / 'RULES_END.json', dict(task=task, episodes=rows, time=now()))


def full_launch(phase, task, root, config):
    """仅管理本次创建的worker及其进程组，不终止其他训练。"""
    directory = Path(root) / 'cells' / task['id']; directory.mkdir(parents=True, exist_ok=True)
    marker = directory / {'train':'TRAIN_END.json', 'eval':'EVAL_END.json', 'rules':'RULES_END.json'}[phase]
    failure = directory / ('FAILED_' + phase.upper() + '.json')
    if marker.exists():
        return
    if failure.exists() and not config['retry_failed']:
        return
    save_json(directory / 'TASK.json', dict(task=task, config=config))
    environment = dict(os.environ)
    environment['UAM_STRICT_MODE'] = task['environment']; environment['UAM_STRICT_FLEET'] = str(task.get('fleet') or 0)
    environment['UAM_STRICT_LOG_EVENTS'] = '0' if phase == 'train' else '1'
    command = [sys.executable, '-u', str(SCRIPT), '--worker', phase, '--task', str(directory / 'TASK.json'), '--root', str(root)]
    print(f'[{phase.upper()}] {task["id"]} {now()}', flush=True)
    started = time.time()
    with (directory / (phase + '.log')).open('a', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == 'posix')
        try:
            while process.poll() is None:
                save_json(Path(root) / 'RUNNING.json', dict(task=task, phase=phase, pid=process.pid,
                    elapsed_seconds=time.time() - started, worker_progress=read_json(directory / 'LIVE.json', {}), time=now()))
                if time.time() - started > config['timeout_hours'] * 3600.:
                    raise TimeoutError('本cell超过最长时间，仅终止自己创建的worker')
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        except (KeyboardInterrupt, TimeoutError) as error:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            if isinstance(error, KeyboardInterrupt):
                raise
    if process.returncode != 0 or not marker.exists():
        save_json(failure, dict(task=task, phase=phase, exit_code=process.returncode,
                  wall_seconds=time.time() - started, time=now()))
        print('[失败后继续]', task['id'], phase, flush=True)
    elif failure.exists():
        failure.rename(failure.with_name(failure.stem + '_RESOLVED_' + str(time.time_ns()) + '.json'))


def full_aggregate(root, tasks):
    root = Path(root); results = []; status = []
    for task in tasks:
        folder = root / 'cells' / task['id']
        train = read_json(folder / 'TRAIN_END.json'); summary = read_json(folder / 'CELL_SUMMARY.json')
        failures = [p for p in folder.glob('FAILED_*.json') if '_RESOLVED_' not in p.name]
        status.append(dict(**task, trained=bool(train), evaluated=bool(summary), failed=bool(failures),
                           actual_steps=train['steps'] if train else 0))
        if summary:
            for mode in ('DET', 'STO'):
                results.append(dict(**task, mode=mode, **summary[mode]))
    save_csv(root / 'RUN_STATUS.csv', status); save_csv(root / 'METHOD_SEED_RESULTS.csv', results)
    groups = []
    for environment in ('E0', 'E1'):
        for k in (2, 3, 4):
            for method in FULL_METHODS:
                for mode in ('DET', 'STO'):
                    rows = [r for r in results if (r['environment'], r['k'], r['method'], r['mode']) == (environment, k, method, mode)]
                    if rows:
                        values = [r['late_co'] for r in rows]
                        groups.append(dict(environment=environment, k=k, method=method, mode=mode, n=len(rows),
                            mean=mean(values), sd=sd(values), minimum=min(values), maximum=max(values),
                            late_sd=mean([r['late_sd'] for r in rows]),
                            best_to_late=mean([r['best_to_late'] for r in rows]),
                            completion=mean([r['completion'] for r in rows]), rpt=mean([r['rpt'] for r in rows])))
    rules = []
    for environment in ('E0', 'E1'):
        for k in (2, 3, 4):
            record = read_json(root / 'cells' / f'RULES__{environment}__K{k}' / 'RULES_END.json')
            if not record:
                continue
            for rule in FULL_RULES:
                episodes = [r for r in record['episodes'] if r['mode'] == rule]
                if episodes:
                    rules.append(dict(environment=environment, k=k, rule=rule,
                        control='J3_PRIORITY' if rule == 'READY_QUEUE_JOINT' else 'LQ', n_eval_seeds=len(episodes),
                        co=mean([r['ATT_completed_only'] for r in episodes]),
                        rpt=mean([r['restricted_person_time_per_generated'] for r in episodes]),
                        completion=mean([r['completion'] for r in episodes]),
                        all_ATT=mean([r['ATT'] for r in episodes]) if all(r['full'] for r in episodes) else None))
    for group in groups:
        reference = [r for r in rules if r['environment'] == group['environment'] and r['k'] == group['k']
                     and r['control'] == 'LQ']
        if reference:
            strongest = min(reference, key=lambda r:r['co'])
            group.update(best_tested_rule=strongest['rule'], rule_co=strongest['co'],
                         difference_to_rule=group['mean'] - strongest['co'],
                         rule_scope='已测规则后验envelope，不是数学下界；Joint相较LQ有不同控制权')
    save_csv(root / 'GROUP_RESULTS.csv', groups); save_csv(root / 'RULE_RESULTS.csv', rules)
    progress = dict(total=120, trained=sum(r['trained'] for r in status), evaluated=sum(r['evaluated'] for r in status),
                    failed=sum(r['failed'] for r in status), completed_training_transitions=sum(r['actual_steps'] for r in status),
                    nominal_budget=120_000_000, actual_budget=120_422_400, time=now())
    save_json(root / 'progress.json', progress)
    return progress


def full_pack(root):
    """只打分析数据、源码快照和设置；明确排除模型、归一化与RNG对象。"""
    root = Path(root).resolve(); target = SCRIPT.parent / 'experiment_archives' / (root.name + '_DATAONLY.tar.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    allowed = {'.json', '.csv', '.log', '.md', '.txt', '.py'}
    with tarfile.open(str(target) + '.pending', 'w:gz', compresslevel=5) as tar:
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in allowed:
                continue
            if 'resume' in path.relative_to(root).parts or '__pycache__' in path.parts:
                continue
            tar.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
        tar.add(SCRIPT, arcname=str(Path(root.name) / 'source' / SCRIPT.name), recursive=False)
    os.replace(str(target) + '.pending', target)
    print('[数据包，不含模型]', target, target.stat().st_size, flush=True)
    return target


def full_main():
    parser = argparse.ArgumentParser(description='E0/严格E1×K234×完整五组合×四seed；120.4224M实际步；先训练后评估')
    parser.add_argument('--root'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true'); parser.add_argument('--allow-code-change', action='store_true')
    parser.add_argument('--phase', choices=('all', 'train', 'eval', 'rules', 'pack'), default='all')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--eval-device', default='cpu')
    parser.add_argument('--timeout-hours', type=float, default=12.)
    parser.add_argument('--worker', choices=('train', 'eval', 'rules')); parser.add_argument('--task')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.worker:
        record = read_json(args.task); task, config = record['task'], record['config']
        try:
            {'train':full_train, 'eval':full_eval, 'rules':full_rules}[args.worker](task, Path(args.root).resolve(), config)
        except KeyboardInterrupt:
            raise SystemExit(130)
        except Exception:
            traceback.print_exc(); raise SystemExit(1)
        return
    tasks = full_task_registry()
    if args.list:
        print(json.dumps(dict(version=FULL_VERSION, tasks=tasks, actual_budget=120_422_400,
                              ppo=SOURCE_PPO, late_steps=FULL_LATE), ensure_ascii=False, indent=2)); return
    root = Path(args.root).resolve() if args.root else SCRIPT.parent / 'serial_runs' / FULL_ROOT_NAME
    if args.phase == 'pack':
        if not (root / 'SUITE_PLAN.json').is_file():
            raise FileNotFoundError('不是本实验的结果目录')
        full_pack(root); return
    if args.timeout_hours <= 0:
        parser.error('timeout-hours必须为正数')
    if args.phase in ('all', 'train') and args.device.startswith('cuda') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('请在启动命令显式设置CUDA_VISIBLE_DEVICES，不自动占用共享卡')
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError('目录已非空，请--resume；不会覆盖旧证据')
    root.mkdir(parents=True, exist_ok=True); lock = strict_lock(root)
    try:
        native = materialize_source(root)
        config = dict(version=FULL_VERSION, device=args.device, eval_device=args.eval_device,
            trace=str(native / 'train_data' / 'passengers_300.csv'), train_horizon=500, eval_horizon=600,
            actual_steps=FULL_ACTUAL_STEPS, requested_steps=1_000_000, resume=args.resume,
            retry_failed=args.retry_failed, timeout_hours=args.timeout_hours)
        contract = dict(version=FULL_VERSION, methods=FULL_METHODS, seeds=FULL_SEEDS,
                        source=SOURCE_PAYLOAD_SHA256, base=_SEALED_BASE_SHA256, ppo=SOURCE_PPO,
                        fleet=12, actual_steps=FULL_ACTUAL_STEPS, schemas=[FullSchema(k).manifest() for k in (2, 3, 4)],
                        training_trace_sha256=digest(config['trace']), late_steps=FULL_LATE,
                        reward='作者原奖励，不混corrected StageII', environment='E0或严格E1；E0飞机actor关闭')
        signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        old = read_json(root / 'SUITE_PLAN.json')
        if old and old['signature'] != signature:
            raise RuntimeError('旧目录实验合同不同，拒绝混训')
        if old and old['script_sha256'] != digest(SCRIPT):
            if not args.allow_code_change:
                raise RuntimeError('代码发生修改；确认修复后使用--allow-code-change，仍不会改变合同')
            save_json(root / 'code_changes' / (str(time.time_ns()) + '.json'), dict(old=old['script_sha256'], new=digest(SCRIPT), time=now()))
        if not old:
            save_json(root / 'SUITE_PLAN.json', dict(signature=signature, contract=contract, config=config,
                      script_sha256=digest(SCRIPT), source_commit=source_snapshot()['commit'], created=now()))
            save_json(root / 'TASK_REGISTRY.json', tasks)
            save_json(root / 'SOURCE_MANIFEST.json', source_snapshot()['sha256'])
        print('[预算] 120run，名义120M，实际120.4224M；5×4096；先训练后回放', flush=True)
        if args.phase in ('all', 'train'):
            for index, task in enumerate(tasks, 1):
                print(f'[训练 {index}/120]', flush=True)
                full_launch('train', task, root, config); full_aggregate(root, tasks)
        if args.phase in ('all', 'rules', 'eval'):
            for environment in ('E0', 'E1'):
                for k in (2, 3, 4):
                    task = dict(id=f'RULES__{environment}__K{k}', environment=environment, k=k,
                                fleet=12 if environment == 'E1' else None, method='SINGLE_EVENTQ_FULL', seed=123)
                    full_launch('rules', task, root, config)
        if args.phase in ('all', 'eval'):
            for index, task in enumerate(tasks, 1):
                if (root / 'cells' / task['id'] / 'TRAIN_END.json').exists():
                    print(f'[评估 {index}/120]', flush=True)
                    full_launch('eval', task, root, config); full_aggregate(root, tasks)
        progress = full_aggregate(root, tasks)
        save_json(root / 'SESSION_END.json', dict(phase=args.phase, progress=progress, time=now(),
                  status='COMPLETE' if progress['evaluated'] == 120 and progress['failed'] == 0 else 'PARTIAL_OR_FAILED'))
        full_pack(root); print('[结束]', progress, flush=True)
    except KeyboardInterrupt:
        full_aggregate(root, tasks)
        save_json(root / 'SESSION_INTERRUPTED.json', dict(time=now(), note='相同命令--resume续训'))
        raise SystemExit(130)
    finally:
        if read_json(lock, {}).get('pid') == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    full_main()
