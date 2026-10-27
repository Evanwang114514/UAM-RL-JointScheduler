#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E0/严格E1：同信息S×AC六臂，198run，实际198,696,960步。

原作者源码和已运行的严格E1/两阶段执行器只读内嵌；不依赖其他矩阵入口。
不修改项目env、原奖励、四座、机队、坐标、充电或物理分钟。
--resume可续训，单格失败继续；--phase pack仅打分析数据，不下载模型。
"""
from __future__ import annotations
import base64,hashlib,zlib
_RUNTIME_PAYLOAD = 'c-l<4Y1i`Vbtd>dYvn&2B18j<Rs;wMnn_|5gFt`;hz~+D5}*MzAqiMXC$1z_cH-`~op~&~>`KZ`swzpvc834bb>8<Jf68BIj-A-~^xS((&%L$xzV>h}fXpui-tCuU)BYvR4nJ{Y-*D`o|2g0jxCnouII3wM|IbeXUx#a7Xny|Ze&H8>VSxP7r+@#gkAM0PgVKlZefi_p|K`&leC@Lz{@RDH{>H!j!|#{M$Stt{<sZKM;j3Ty^!va07U?g9@BQ(Qe(y_f;TO?z`4>?H`S{&${qJA?zkdGbe*Wh^{rm5J{Kl_;@xotx`SZW}n~%TyTkl&w{rhiz_U*5H_FI4T;m?2dfB*8YOv@GY{pW4x)4%_VkKg$I$N%~D|L>Q-_VKsA{oyzM?kAStPe1&@AH1J__MiUSXMg#hKL6f#Kl|aYzOVT3-LL-G&ZqzQhoArQmp=aBtDpVpuYCCJ|M=m5{f{4;`;~ul<kKI1>%(vV#pi$Z!w>)bm!Ey<+aLbJmp=UMm;dD--Zy^b(|`J-&;RxtAHMW=pZ@S0pZ&>y{n79K?T0VDU;fW;x4wA#&)@s_weSDv&%XcJAAIi}z@HfV;g>#s<F7yb`5%A&XW#kkn?LyU?|<v#AN|><Km5DTf8}fMPyOt>-}vn7-}=ej?|=RD`@j3)dw>4L_y5sXKmO^je{uNMH$MEw|Nh0{OJDlzYk&JbeEi{`eg2RC`0+Qs@$q;5?nnR6pM3sT-+hPn$EJSb=!Y+T=SRQ#oiCE{8(;nK)!+K?2Y>wWYyb4N@Zl?8e|z`goB#duzxmc%|Je_}{n-z{`_{woP4fav`}iw={^6Tn`LV(^Bv1Rqm6Y=rU;mr`@5itI^~bM#@57hB^kXxhef$6Z@ZJCN`9FRCv#-CS^7{ki<Nx@xkAL`^pZ(V#{OCXb@lW#b<8b`uI|Gx-7b*F}?+lO+U;3Xv`kn87@x~wi;NQIdB4(d``>#Ly!#{ba{3ikV{I7rY!w>$;kL|qC_47Zc-yG|c!$A-H;HDi8pO}{G-26`@+je}(Hyzvi0yqDr_57cn<+lqM`cqMH-1ujL7yk78cGb9;@=s;c_FPT***z!kixT)(!IFIAr-I}C>*u*|?fZ&<`!?p^zjTvq)v<mmdgjrVE<Y0;<*fb8miL8Ao^LAN&s2fz-W)~q{+%lR+1ZzFx_QxlCTn*;x=HTO6sGm>p57!yliz9msV4cc>p%9wk6q3CHy>#2UHU5fR{i29>$mQr%_U3o-jL9~h&S-b$M1ggU#sJ{|Kbgd4`2PukH7QhZxsBr8T_~`KKuP&`|!Qrf8+C=>yLl@Yaf32gEweD{KlXD==Z+)=@0(HyNnJ8OET@l;p1<9`}05h^2hJ}?T<4LeDc%EdqeqO%k>+-{;rxI{qk2oeD`-g{pUaW?CZbxhU&*Z{BJ+ee^Gq@@cVD@eD<~PeEgkn|D-g3f~60?_njYKdRPz0*`R;ulIy`Itxx_B1$M|HJGc7{X~SF$<}}CLiq~C(7#cf)%pH<+8sMUbLcaW}K2}AzDlv-|!@#3uliEq6qLI0t=f?3s70aXEbrs0ATTix`yO7b(Q{s|6v>JAY$}A!l^3j93L;R$-a0{W6(+cg9^fc3#nWyFJBa+;?ye?hnjxGc<yI5jr3s|2v4Tk9MW3;u?cKO7&gN~1ZaC0{dhtD%PE9xY*$pP@<P&8YnH6raNPB!+hC@D@at@SwBh!ac8McGG62WFj!dwq)lrwjG)j4Io7o|^r36B~>VkC*p4OpdnJ+)ne7FAQqt)apB@Xp`SH--eO|e!he~{LC#b$-QSi^)9l@9X4eke<7&6*<lYu-0JxU2W^fDP;s~$j{~bJ-_I0obaHNHb*onC!<Ag%&`2!#*?r5sk(soU(xkZPN`poX=<edq0iWN9jBtSB-eJ*sfc-YnJ>mPQg?YG6IvadyE%MS~WrsNYsx>!hpNr6x*q#Te9|R?Wf7oEBwHU<{Qe)tD<ah<@1*hW$^BVbR$Ee)l*SG?}Np^e|LLpZnmMt#Kb0wxzq-!s<$fE1}{&n=9N#sEt5pyppY)*vkz4B}+Ly4IQ0Y7y;y<2EENK(4CV`mam?MivQepJZ$<vcxtkh>3VS4|(RPGY%y(Aqg_2Es1dx;AOJ$Fh+o`2z#5nhr`_TQ1FMrn1Ua#nds=eX7;9leo~b-aDZfI6~^>P=$n6L`PzTaG<N9uu-O4K2VJkCPXYb?<ln-(<bbw#jmi=JI;kv%}jIocJ<1F(ekRkGe{#y1H_G!PYwGy0}|95*RarA%1p+)1Kmi%6v%dVM0k6FzQF29f|s4#aT+=b@pv*A7+A-2OBUPR)v_^-4@$k!XbW#$rC+_0MAzhYTEQErB6^^ABG<9+YRja+sYVvkmLHJxl=qD3)M6h!A#oT@papx-s_RYGZ)0_4|8hF-n5x29EA~)GGwpZSq3I~<aOXBgn2@1`QRX#pdq{VF;#Q&AtgQ)|iTPeUo#B`agUDI0GUxhWYT*m9BP_bIVJy=1I~p&4${{G3-bFiqQ3GVDvPHzHi%54!_#ObIOz+91)5~`6GPxR894=~yDL;I0W*n*XM39TwbTQHfsUUzce4bFX{V=#QLm@au)7q$ZIHa|<ziAAUiclNll<2v@k$7~Q(~R2OT31i()te6i&w?urKQGt!yq;V#Kyl9!LMnn;x7vU!*-I<oZqvPJFSp8M0u}`Z@J|QzQmW@A*hFLNbWgO5W=$R?&#$(PhK-6CaPFk-fNJGASSL4NSYTHwkDUlQjHZu4dKi1J0qXf3t83?(QM=bk6>(2_jXs~&C-j;i^7(t3;Y(?{fGd8w;0z&Ms-f_O=e)3fVllbEfR6UivPI&)(A7d#dnOGt9BdElY1|MVOQ|uw75dnKc5px|uFr3@?YvRIbDKx8D6Hy*(z23wN=HBv^E79x=Mzb!!*!O4cTXr<cd67x>m>~*&c$9(;I=KO=X-rGP3OWKq>%ctSC~(Q^ks2zXxz&L9-oTX6w42x{2qJOF2YrmG}yWaLJKcG;%Oe%E#2qhWXP<X$J{~V>{BQsDB!p`Fpr;3AYXk=cqSt+9)teqlzDiSbhNxXYbmYTjaCctkKR;icNKc~90|l3%FeTUXU|seeYQ9sr#Sl32iaCOdt&SDgz9dhSs7)~6L?W#!D!_-(!Ezh%fZH6F51~aRSyp#H<)fk(d=OOl(AnyVcc)Tk9T!hSPDo3@LtEWJ1jFz8NF#VDY`#N*5%DS;X0(7HX7o#S3=f%5Ux*vjDYx)Wt~4(r`VIfxmaa9+a1~~x!e+%S_{>p3jog*>v%s;865Yw%dm5!@NsusAkT;+hIc<<TNI?^XALh36L|O-dyJ8#PHbd)k;`^XL0$x>Zq<BNGKHBmSPM(__0YX~(~c+EJq;anIJe$bhuOojm2Z@?)!EtYi%<C2hWRGgeKjuNOS=+90fO;U?wMXLm26(l==Z+3MzXp0oM`qKPNFFSNzvMX1?PH|V9M8`1a+)F#MHOl!I_}824OSE{(eGRTF|QoxpH?rsUfahXO8zVWG2fgczCV6RsP+&JLfH2@(*SMtn>48^QBxZy0u1__mJomWAb`oitSlWAGZ0Vv)6`AY|D8BGhX%#W<Sa~;Wh2TujZ>D60LmFq_>>49MEiQezSKaXr*Mw#9Ys-zWq{&plHAG8t0k{+_my0VUWLyNIPDa+m3=}$#cEp)_H~A<P`*DkvRrts+F@I*%RbculJ6{_4K$+`5byGd(?J==?3N5Y$XEKyINSUpmKb8<N|xRO3=n@neRR7^AxA6y?Nm<^+ybcFPC@vFn5-%i$#WwJTvUd$PS*2oWYJgW{buk%Bgbk8m!5Q>vteqZBt~`X>IP_D@*tcIawaUZn&ZfIzF-q_&C}iV)?Ok)tcVXWIA21tK~H~YI3jHmcegEMxcs_HK3V|Wc%H;vBPVR?(5m%?QtJZua8`QA{oQFe)c0aYf@XkVH`HPEfz12W4eI_)%V(ub9uOh-Bq`o)_d3eB~9+yXy|LRp%|U0)|w{cS$dUywWZywNl2z}Lwt`8n_ezw?72fCw-fI5#P@YU_BdJ_t5I^mPlcjF*-PP4SBOl0cn5M*ZPN}pgO1h(?>edUYcnHvB=5A<)7$2*&9qt@{Jy%4@`BXGGlEC)9u05QYFqBHW@&KC?DN;vB^S<Tlgg+mm#%xSge<!+X(Yka!9H!>N)d|s#lAv@&}yzg%2hm_stGiCDlb<p^7t^{Wt`GO7zh`_KYPo(43^cyz3?E~D?)?yOi5z8liAuDW}?gd_`K@<Q*!IFm)i8UQ1tB1v$y+2@2SSE(0}2k0jzh#2|?8^4W^biVGn8<<GjI(DjrssN+)j<htt9XTq5SA#Py2Fq}SNmYHkCsGoh7C!5V)i)M+cu-pVV)WJ8bY@>zQJNJ_Kiz8gNVAgciv98iyIvZ$8#s=DU%Xvfwf^YjcB^jujm_Wj+l`%?((EyhZrilGN|BD05=UgBp}G1}e^dLPQGQ}$^*w_#j|ZU${aY(mm<X;^FL60L3>HHvNq@p;F~J(%BSizO;x-J;*5*5SE?6U4JvB8JBPlmPF|lkKkvC#X#-7=-kVwv4x1W!d)kt!OP|V1X8aO(}%dTdpai6^S_B$l`_zdbRE|K=^5o<&1VIvr+;XSq$zw>0~d>$+*0PD7bv?8IoL0JSTBq6|~GxP6Sl)7JxTp^*TCR;f1SS99=Sxt%~*;>5d!wVZ+IHI!efqgOy%soM6#&qT@z7XC!!eBrkAK!V;dw<j(ST6pu4}=`E4=eO0@wL6P?lfWwPXwq1zn4(#on+rAvR$F0GR8qtiS)>b^F^aIz<43~Mc@c`~td)3)=EU0bv`PMB7uTxcik9RJG@~P*P77_z?rxNI$wCV*WLU}ghN3BVIP#*YUucMxC+&a7<dwSE*cJU}hmuJ4XmX7V|Y&OcRI-<=LNZUqdBJ0S@Hjupomf=@9_^K9OFWb)6L(`kom^oGBE7v|*U$a&&*v?wr=_NYMPcw`Mp|-Qm&`kS!^1;Gl^BiiMPM2xamlR_)nHls5Fw_dF8^y()@Z=Yi?h8rwzBz~03cX9M%1HL3F^dllukC$7A&)As#PF)e<AIB6-4fo+6{awSg52xv`RHj<Hfu!k1nrX815KU`e3!<3VCq&Eq}?t-9>KeopXQ#ura@4}aby$0tcg^|jn}z@y}bHdsNZk<;`HgBjMHNJUdyY)miA~7^WFp@`jb`%2UkX}D;7e&MiJ1U^n`50mE;S9Nw;P&+uGnp#BlU*RC>92g52YYdYUZ62GTo7`+k&7Ua~jx!uOhDOV9FJD@9SSGF6$1WakLeWqLT>C(oHDP%1aT5HE5_WFiTc#wR>8E5hAu(vZsceDp4n+P;%j7ngR2F4lSf6{6!^x=JtBdC)2!h4G^(oNrTj;Z!O_cr50V@nA6PWb6AKb;#DU<CWU$1eAs@ANehAHiy$;R9(H90)J)37ROYTqGHzp+6G3M@wq&z<=1aUi-q%rPzK8xv@{-v;qeKnh`ln%_41Vr5t$s0vwqT5;$gGsB)6K952jPUIg@MmNBcpw3-}2pdR8q)9;z^CpYb@qp`O5N#GWR&Dec)BWFKTq-URdcWumCpA)0Fp`Y9TE>oM5b7u8W`WL0SXS!OS$52}7Y$~(q02aoq%ny1E%QK04aD(LgeR_M+XC|dS*+a5KvmE=uCMozq@BrAku2x+&E`JITYvu6ev>nFOH(HjQ1yI^~X@=Io)E+doo60^szx&t3%0J5sd#8~L-K`(t%0*&{3=OhsfW)W1z+m+z~aW)QDFM5%4dbxTIu!;KwnU<{lTtgP&Cf_q05}d5ojIt!#R<z9yMAfw>yoa4K=rPai`Pa0D8NzGErZuQs_Da<l=8ZFV^mh%#IH3l3?90V=sP!=aUYQ@g2pF98o}4I5kLEk~r9C8dd+XxdlfLP&53zAiwL3HaX;*#9mST+_K>fyiz8fu%T=o)lc~pHcx02Ex&EaAI0?&4y^<k+XVLZ-xWNSb#J6RzYR}bc{d~SPVe$ttA2xP!$$H(2=FRW#CvN>1*TGphKQjF)pjC#11TgVWZR*8Re%X4MoP=)jLCU4{Vh0j5(JtBxcUEQNgv!mDicE-3#x>ZE<!H796NQ-%n)hhDXhq*biXx7Pd3h-e4$xgA%oJac;_)&qU^{tIxs>(7_8w*}H2B3Yw4eerCNzz1jBeRU{#@Rl;y}+YpQ31+ty3!opGkqmd(8_?1$NPK5uk`WUCXL#a2|wB83i{$&(urf+8eD5BYGH;JSM1C|g1}OY6;}Rg8uq-rUt2`3-s$7daRN`Tl{|CHtmP1Xg{2`Y6~`%JH5xe=Ov=nSi1o~Q2(8v=Azt`FUg3o2i$52omk^K5^|DBDu?;iZa@!iuj<4H7g0Q;*JuWTbz#Obx3wh2@6Oi(`oOS`qCz56=svn=9>H>93w<pIpcrcw;r9F@=SOr{z4PAv4R=v$03Ud-lNNfY=XjzS^icYLU2;_>rJ3rq`sykXxW#?h!9RaGub3{?*lLL43u0d$^%?jaNo4)RB*KTs*Ybl+{hlA>QzNX!YlH4FGZ3St#%Pnrs%<;5NX>Mv+J1TG&qGk_*eoN>H6)8=)esonNy4(kE5~|m33uiR!t@q_)R3Y|`F&9_pjA=&sLeS!r#YNSNC#V=+=-yOE)x9@3FSyw$lfdmbs6gSgo4<}o!v}f4g}L4cqRe6;KkeNYdw4DiyZRn1YNSa9?<GX7mSWF$a@5J%IswpJ0E=|gOGe(XTip*U_G(hmXD3Rlm5LFG-(5R=>3WcPD#sXBPF^#Up_opNSn_!LwiB!E{-s%%OU+H=u{z-Wf<18;b?X8QC>>Kh<au$nbDL@NnAh4If_Z^bm|J<dD|Ig8%n;AR{b^i;>rEldwIP0AnuT!E!>@OLJmTicv)nr+BUv}GQqP~KnvQI)BkO6KQAb&=9A-2IDayCA6#^?Uk9(tH$7aSV>^HAfbLQSgIrzeB0j>^j0<s|9Tj=GG?$0A=D9Kavv6R(9jZV$k?v^PyJ>H;uQi&dy^~Ylo9DB>VH(<4rcY@4m_I?s_+uR1*Hcau9<e`4P*cvyM>*dqmGq=sE+~ecz1Qyd5HMe9ZTc~w6MBwrMs^IHPTWSmJEBnG5fit?>WP6<EbFH-2ODa)8INyuwm>-#S>OHwlIkKkyK<sH#wQq_!eKhaV*U~lYqHI;U)F6+J?$JQT@9&M~elfuTcDW6LD+$#R$#*k3B)qeV*e7pFFMLxgJSoqy<$KE(cbyJ?l6#0Yan;nm;80-#KNs?)7mf8Dx!K;m9#(EHq^X-{GAZXScH1L52egjrKgWgs<j_}VNuAV&(BqICrh{j>yUlIcdYmoa*k-18n^936%Y<`ir-r-Y{f30KX&U7tfIGh!M4ga*1ew)x6KwV94hu`Rd1FsbYZNy3V%wSzSjf`kJn$~pbC0Cgn35cflsb1ZxpB=~qIsLQ+vRvXkv3&e)X6Mvx#>$B)S!1)49%Qyw}eqPk4Lnrv%|8#F!?Ey=`gbj1BU0aTYv2$ybxE#a(RTk8`;oY&#JOl^n4uEl<7%7A^9qrc@}OlTD*WmtjfT8U{tbj)v;Simnf0#L4`91+oR<kCZY(?#ak*9227uC?cK<`W$kXi#pRwyesen4mFo~{J+^~!D2-^YL^e0UL|Sjcdk3V*<~S$TFGe&)Q}?Q_<kEb>H3x&ohB&Rpjcs>tS1(q%p6on1GoEOdoqz1}l5y#!=qVOhq(*H}cdbydQFx**uSAu7wKqC8YrB1+2rt^3&T0nUQJ59O1!|`YhD38lj+aXARQIiiNndyRwFVY${NlMwOB0=XVDfNJN*7XJ-3=Rj6SeqU8C39gk%w|@cTtiDWpU#tG2R_=+(v6Z_u;a8;O$Wauyy5dB6xGOYz!Xj*#z(8GF|sD#kY$tye)-n;h0nlgaKSV2=CfDhoUXv4VKBEP^nNQ6<VRWJMs2ntQC;k_*$-Nkkm2+ojAa)gzlkQwzR(Ax_vobk4Y%=#Ax+$#24m?Z0%P~MF>ZG06fhU`RI|=oO~-Bg|xiOh&Yx#-0tEWrq@I!ii05Aamsm3=76=-+nUq+=1ua1b=ujC>!%Cr4T$2r7yxV*%J^}P2SZ?e4TamNIncXA>*my{Mpxl?vw>Nw5fM=;yVx^IGuH!(<^7^k>5zG@bsuh_&EmRairA8#F{-kh4(20*lS-DcvhdqLk=>o+-MP&Ly9Ize#ic#g4j#w;yhHY!L<U-&Mm5+n#Pn3(uzUyKZ7*vli@45KaK?Dph??&$<T9D8o4Ist%*qDWi1hdR&KC}tG(16olD%>H`s8rD^u`7hM)`N%L+&I!W?*1q&aDx^^6u&&)O*n8+u(V%N3rU50%D~Qy?PjtrMw2LoYNC=b+_(LygyetcGY0JPaVK(V(y;O=_G?&fLG7D)zn<;$2-|3-}H6V$e!v7Kb5D$v||cGF)&B@QRtti`EF3RX}&I0c4a&-0a=sUR2LXu8+F6OMMcKL#{g|on1#!|_6<15)7hM6m$E|+Z7hg-?MY^MD2RxEM=J^<YHlZLjZ3ZmSn5!7twOygE|_rq8Fgs<3gpe8=j^iEi7LLD8d$?ILWO4cbUBAxuhMdt@3+fb1RC8aLeHFcA9+!6gAD74nqARNJxP}3X5Jax7xw6a7r-9a7uGUTFIP_k^LV9BkLcY8N502RBz)qn{Mr~Uh`rQUY@6J!+$`YdSuJk&$4{P{>58!xP#vxH%kCMycW0$&IDWQbds<~5`Fojd&ThGRFf2n^l-~o<O>q&pP=WPqHK`$5(QxMDCO-+lEP;2hd4Ha{g<C)41$m(79m1_w(A3f#<D0oWGJ%V-`82l((_opbK+JX&?;2Nkb<f0JZZYn+POI91Wicb5aRk`9vV>O&GTY9ueExvm`tTena+}_*b=~xnx~=vf^ZFuE_)gA?9TnZ~=xrsZlTmJ7AqOB>;=9g8>>Ic{GVZ12_;PsfVe`(S`B+q+z!YZrH5KBDjwsu9v?Ch5yf_Jqg>JW*jB|$_0k4>)sq9B2g4@6$yE*Z76<L$!jtffH$^x7qnL*~^P&MY6XXR?q5yXXNYas98m7JsV)dk?2GTI*VBS_d9CF{B_IDBifqOR$VrtmSbv6>xSKAs8~I1kMYLg&L;y{0_H47dRm*pN4n4#C&^b*#hO7F6%H%8?bW#1$%7N}^;g7#4*x%K2oZd9f4@js-u&sAe=0Lt3hto#f&Vw!;#FR-sj;CCmHK;JsFs))Bhypb<4t*6r5(x%bB^>m;Q?V)YZQW|6L!v!t5}MO=(@=*qlToTun<!>w<1Cg2<#tM_qxSe=i^n5QswtqQ<p-E42wt1`%Jie3JF<*v&lh`Ovi%1SyVPpggE$9C)Mtv|l&?7r5jlD$qfMzW`rr2F!rht^`yUPkA=XVjO?=qbq?C-dHXTvXS(cs^^^)K^cqAO^S}J`PY+Xg|c+RURzMxj`f9THv)0%zN*eDW00_n^$M#1QoKgh4>DqaO)j6XP|hHtpfn6qTC<n?$+WyyIf3k+7EY^k*W2oSR?A4<HKA&RL*OFCwMv)D+eTc?Zarc=-_nev{BZ0{^*2pJrb@YF?U%EA;4*iY`5q~o0(2cvZk4po3%Zc)RuIck7|u{${X(WcEIFH4rc33_^Cu!P+3gImo&>a(beM&GD5JdB*XG%XmL-n4ej*+o5hv5`LN7Zb4?x8EHyu5YtvnNDb<8+27ip)`SY0S{K^bfg=E8H9Gn5g)do&y5evWs(LrL}!Suzju3ZYlEPSe_w@h#TT;xj7l$E*4ag1H<?j1(~h#Hp@R3TQVd3=HG{&n-%Alfo8p7Ess@5kj`IYV_STVb%*kdXn~due>1(#rx|=y6ndp9kR)KS90qD`0QFxh+aU*c)Cce>b!}>@M!J^U{1a1;&f!I_lr)aV2o<#~{j=PeQjlKTW$2Q>x5}wp%YxVpQ7KV;?_lgw_)|%OL#j`1~eUKexuyq1ij%j|8p7z3zh5>$$D^l$|bZWNu|T8mq@-y`%OAlMo*3DpxLK8+d$`i)2ft9y5DbD|L+J$(8T|y$NwVB1iDJ`((sQ71i<?tUp;&?R^`hvPe9vy>=3`X+twRwe_HGw0ez+Oa)s~1F9X6<0vW7Z&rHEjb!n>X?3+xj-95b#-If*^FzakOYKTW$kk(hm8~{<dxp;%62DcU5=K_~ViJa+H0wN+TyDP1&BENi73Nwjt{KN@?b+~Ilw;io>f>w|J1umqPAx-sd?4KkHja`0SYD7Fbp=)VRw(aE?l>=vn#g;-0PY%nSo2GN+_;Y}!NTHkLTENO&xIPRB0sdB0=ZxdgIkfCbr&^n_`uoXs?A6qJIHnIY`zHqmB4N*XVe^BGE;ib)(T|%{$i1)W|woot#IKCPN!`<_FnCBwl=EyxQRdIE~qf-VR?8~@l3#$=qN*2t63<NcO$6@ZB4Sj1R4Xf5W}}QcI?%nY&7RWO^M~yO2imj!HpSa6#<NIRhpHu+MQ{_Rmye)WlEH<puH76bvP7ngOt}QK{-#4LL=-$Vr9}8$ihgOxY%Kgb8F|?7xtLg>Z~~QS`1mXR`X60a(Ksgr(`8c8&dN$orPtIA5>p;a)uf+Dc`S78wYxLUvvid!HUuE&6uo3P2H&Uwruvwhf4w5AdP0N1A)=m(zo2f?~bGU%qciHQrIK;xCIp>qGLq;Xh~(z*M7Y%07P#4c)3gsF1CqqJ38HrC)m_`c}-t5m#Ffjrvcmb9{pPbfTWlw)fPV`>XeJe$RUqfYr9geWW6)DmwJ33Sm1pxRR)hKE9Tnos5Ez^DLaHVAJ3izAngTsOP|gbRI%kY4%2ssLRu^06>%z<maE+Dxk0G~kOBIzSB!h+az1$lCDS7W$7oEUYEYT<k5>IT4i}GmklynZsy5i$eYyj0Ekr!5mX?3sQ>39F64Pdx?bY*aU7B&)(9GDI>oAARb2fFX^w&2-5UVXPx~#!XrmNJ&@X2JbTxIIJYtC#n0Itd{?&P*V&6Uiml~HwLkZTjs@thd|uYLgI)->Bngl0R*Sb3}O#oMV{nCgR!v7OvAUa3?LR9U)Bz5N>i*Q}VF_m}fvj<rs&u$~Cx=eSlcFEPk2X=J<{HM7L2j;Ys$O^kfzHAwoaG1N$Qn?Bw`;Hf&eI<su)jWW+ZAaW;&9_=>;o0EBVc^R-oqw5q;v_YN^&hDk$fVJOW>9-01jSki3gWb^k8zc(#2M&WLwJ=wG1PO;QBRFMDr=igrKWyzzh+R%7IG)`4BfKK3ox(ZOxY2DH%n!$zgO#%Y-%z&M#sEbc%uE8kR)}u>2_81NZrv#l#^(aSscCy5uTG8Lx;1-kDc~ij!`Kk>kyCGC#yKxNDhkD0bd%!9Ql+skjfDB#tDhbWtkv=hlVJPz)BX5lmxDs#<qWQ`R<=$I8#ONOz4F@F%(~~B*I0nv%)5d({&s&dLg_}sd!w;8GUI9GT0eq^n|&m?R`~^T4(Hysh03BvX1c)4tw}elx3=bTqC8G%JAR?l1zD8CSbeCkhN}~!{K>v%s;64BuZ=EQ*`f6Cjh@0RYqi|Ty){)cqR}Xq)9$pU8iiUA(EWqIM{@(&t}A_xZojNo*W49DP6884DY=>-0tI|O3Kz4AR)r}s*yGU_cL98Wc$G=AtmoNxQ+y^nUl~ep@iiX1mHM+SWGfT01qEu{k{_dV0XR8Ch0pOotVA7p3pa;W>(bqwPxzsU6tU8-1~<JVOYf6YrN-Z8$bLp8n6Zhg6^&dJRz#STiqo>m>tU7Tm}@sP8oq?gEK!An>R-+N87Mb_ObQi_mqrXK$6XO$KandmF|-ony~ifPgpJpYd;@*IT3Mx2qkX+W;AOCiM^$GNjf7os?)uwu5o2cMlwqqdtjuym=e#Y4&Do%e)%mvS&5Ru|FnMO?&kF`$7LPYEZr{=DJUAM|3LfR_W6tN>^&&hxtldk-LF55kCd+-CZiLk10!OBm?U+8}`L)di<vcLBCgpAoj;B3W45o@`_qgTWNBr@0A5O%kT=6fLjgeoR0&6Fl=n2Q0GGS}oj5kE#6$s+-V8febcAhPtwfUKRs;ax?v(f%87h-;4bWMHL1xmQ4>{zr;C-GyM=(9>FFM-nNUC6t0*M2_cnWhp{^wIz_aw^>5W7-DGo9cNdtP6ecUN7f)WQ|o$Kr@dOwsPS(N6qz~OuKI0PUXg|L-_-NU5u8SoU{<)(bD6KMehD4&l9@A8Z}4q82#F>pmVl%=;r{@d`SELJyp~CX^4W`Hs%($P@*kYt+GvG&&Dz5r7iEYJ7<<1cb6^+FuB7vne+ux^T)i#4)&F1-hYzC!4=e+z5aH2=_BLnjR-g`USam4y0R@W64;tM5Q6Gc=w^>FP=qdWZ#>@5sS;C&vop*;t>n`!-+^ZQwkfH~VN!41w*Xv!09X+27K~9G0p<M$Fhs%1E$^%Ib4EEH*`t>!?T_V|$(?dVt^SmG<!qpG<=JePpK6P_P1)0&JTx6Jo%nv?`5Y})$$Y;aQFMFJne<(|(TFzZ^Ww=ZdJ+#L;3Hy9=(O-J#&TCLd&zbOZu|*M`8dpm*I6ewx_38vxR}2#tIKRWMHZz(M{mYZA89oj7rm@K23x5D)}}tV)$dX@oL>F38I*eU6OL@Xa{ScW4w2hCO_$A?^ZFjJ-R|<XQV^h+TR*rldiG{A(e^#HOzc~`(ea)z;<1zFAJMssGr2;)T-b@~e9@om9s^+big2*rk)GnX$ZlFE;YPOFJHa;3XNMrhnM6CH)mku;BfIU-x+9fV;6aHgjYkq*){y$DJcqW`;VLt&NK>a#FA~yjQOPe0C#lv^u1OHw@XMh@Qr(?7AKdRndt?X5apP9UG%(dPsnwzdG47AIF_UF3(`v)IYYu6&cI!$Z<;{L`G=oA#L7tnFJ!v9(HlJZK)TN!-+?!+(ADenoV=v{<MG8ctpFq=oQ!HoL)94I_Mf0}U1Wu-r1dN*)#_`O+E@+$7?D`nDYU7K#;fipQS2E3msQG3h3H9S>ca=HncCHsjM5&m~)M_KKe6K7qj~B;)OQMlBvQNjW0tb4Le{5Tvc-;EqbXi$HWrg1`=f*WZVGhCwXeswgp~3E!TY~EiaVtWDBj2i_=S`wt8tp_}Jaq#I=X?sDf=*T$j$1Z2Z^Yf6wOllcf-g`CSE7ZxSA%AqUcFJc2=y(b9^-+e68UA-hUvt)gZ12C$FX3{9O-ok(%WR7)&yH;#-KHtN4w1vE-m=Va4L^P2czErSc7qSwa2vsCA`qmI+N`*=_-Eo&;2}z-i1&-&(7Ot)KH7LOA-u9GGZdJlV;2b^xA`i{(jcykCGvcF3}}Z>GLMLfoC?r=E}wzy7}8_Nyy`yv<!5H%y3uYR#sX=`l@j}nyq_gY7$#Uou4$y;M}}azDlTEe}p^eB#>(oX=&Btn+B9GyTW2`hIF-^4Kl&OWiM)mDl}@7`kNv=$!mMwVtLWrw&Em8@v+sbN%MX-UOg60U73jvM2zb1S?;geAw&?pD;PPO*n${xybqU}4jBzZIRQ}#+5xNi`r5yxWw$HCq0l(JBt9&(8cyzzcMm!8l69@ik%$&=48}29O0K+4?ep=du_0dDC(EV2j)+{0uVWP7Cv2~#ZnB~AZVrK(=t*T`X2@f=k`elL986mQSKH)LX1*I~C;lv1Hw-Kg9nu<Bp|KvE_E)i9-9!a-yr<RUj+taFI~~CA`ZU=Qr~IgTPR@tj2J|yIOxnj|=sau26^pyf_I3QKPO9_w8i`*Ec7%qxad$&j>eMFeouqf5CKBcf8KA6Bi+MzIk{+h5;Pxa|W6MkJHc+g1j(sIaVIhs`^+iZLFf7%R%ii24Mzq(hJ-93`U}@zwi7#eh@iC=p5)<WaVg*giT|S<T_LGa4yX&W6GAsAjxn;}rCw!L%LXBoXjg}Mos!w?o;BkSMdN*~I1JCH7Z|qad%@5nHOc+P}84IUlVOy#Yi*fUH43^?NTGcPyYT?yhPNg{1nl>^TmLseMbg@vI&z^qa+#T<-_p4{{GzD%gUI)<gY+0!Smuqnnx2#RKfm20*zhXOQr(WZUTterCy&_hW=4LLhNl3&had#SSioVw?MAZew+L!JMDy&nHnsaZCBGY88+{(N(KmP<0-V&QVM$*UmjM7}KPHrjebl|RWC}j&tWqM9B%FtVZ0>Vt60C)x0*M%~w)%;Ai{B*}>3|fq^8k8aNOP;dZxuC^|ExhjEjxJMS3<8vZ@mAgd%*EJ;Wxa6oeEfFurR_o4CY)PtzkucHP2{DyR$J0LdioS+u`7$6yLcRq4`@enHGO<r-#`$ksf&#NT(@x=RM|xWokPGyLp$4nBer<=sq-1nROoJuLfhWzK+IE?%c5a3!IWp&*4W3T!=TEEpbrs6zV;9GT2(J?aW|yHvZl)FGbhvgNmqGr#YQ_UPAAt>zm>rZ&pATNn!nl8yPbd?vpHMLgWZ>3zLexQ38mkZJvg0veZPVt%-|dso{a~K<{G&o)aF(ZGOTr&t-zK+Yn=;R6jx6zvzI;7yQDp!?QSo>pWwxFqpi2Z_mU2unN>D3Jc*r3mRVi>c4vB|@27tLx-3t2qlURUg4^9x6$kz13un+-vHsrhl2JXp(pTh2do(KrM)fF8FGi2;#JdyzraS<bw0d^mz1+)_$zhR(+0%fK7b`^k;zuuraF3-RbxO{OpIOIl+sv&lqP5V6#q#*B_6`HPs`cbW)3=q5(z;!NsGq`$jWic1#-Za{!86K(n%LAs8p=sI8bfV1lrqQ+OrT1JTNjr;K*<$%(4C(09!MtFfH^khOj4Ja)}zt$p>z4LIEJNm81IL<BZEV;gZs(}STjuTt7LcZ?MDAFO15dB_cvU-j#dJBHku@@qfwuo0h~LM;<0OAlw)CElZ%nl%n9@TE0m%k00qTncB`oUFvhj;>gCHlc-3vCR7PAI#aF*=^qz9&X4P$5)9;z&G0vjPaT^BYI{(Vf6~S1TS>UqBapdGaL<+IP?jevJN|Rb+mOwbr--}gw^jI#?yt2bNzX1kfZqT31OZD!i&|0|R#R#TP>sHQ72G#WSy<e)`s}TdAGzmh)NP8im;}+et>p{l3t?<TPn6Y<mnNn?kGwPRz=XcU9TRyjTTcO>0V89j>kJl`hINQbi<PGjV)SApY*@?rJP=CK%JYdu7lfBAl7>-{Kc+GQZDUsq?9*H$TU?X~7jWw4t(1j@?@4omP<)e}XSwSD?>r1k|K&|B5vBXQpBT}P}-4JF-mFhLQ7AvVaf(2R!Fa1tWJnxoStJY-g{1eVY2vge}5|(=ETQL`$t#N`3<(#58X#%j>D?)>{BSF5Xz$+40O{t$+E!glNGog?I$ZNxB30ZiZu2|bD=J10`ZEa<dq{QrL6vu_<Eq@`AbkCKfekcz%iMv&K$W@&^zGCWpytxs4w_#wUIshXAoOXNTzU=n6Ep#5-y4|Z-z2l1w1vJ@I<(X1~xX+PibSYs&h7}<O$_EV(m0+^%k3fJ5*$e<v$HMK#>uY)i&t$(bd06DL{bpj_^?d9I5)|zOV`$`b+Bf9v3(Jh_(PFTBZ1MAw#HGv)5xBg)%1BD~m~%VT8=O|zhI1e)IS8HvGu3ZV$b6)3XH4-WDl-r(7^v#Dfi>z*QUGmK7+}JXH__x5x%?|02ic{I^95Ch7YBzOu6D82%qH>Zc9>V&qr>vVPMUTZ%m=UAXw1FpZFi&$Qtv4Q+u_q(L3xa7bxsgVmRlUuqD1#}9_06D)9oUA{Ju-AYQgc63umYfe@$ywBFH%_$2@RQxVzr$PHP{{gU!9YWgbC!H(2x7OFQ+u`crO`X(w{0o6|_#!n3gu_4Wakdki*$&R3i=tM)lDd!*TY%&!Au<ebWAW(I_4!aY$E5FN^?FV{fx1UK)yZ4K3`yWzE!r8jz~qh@x|q=8V_u(|6aePq2K4=1T!JP~rDA5PJGnfC0=lw0G=ni4CUDlyMZhncj^HWuAP)k3<)8YgVsTn@t$-kW1=YlP>tc1H2-?pb}Ay?$@G5yg7n*J|Wph63I7w#D;pQ?B2hlFIgafw7d^v6~y5L@mP2@|LoY_yQN~R5X1ti2GnDk??uf-IvNn6?~v=VBQ$f!~@dFDYE4NXW1d=mzu<FoE+Dz-Cez!WqLbmn9@LzQKQuEUi2_~W44`}@+ADtDbdxUfaca-pR!l2;ke?T&&I9q2{cAbUde1A6wuQm1F=Q8P4lM>P#<KAxFezX=GkYTJ*i3KX9nMTlk>o1hVE)Wo*~#Alk#0g`W;eooN^$;IV$&JWK*4O(G~EX>9{_kcN(g`#IR88A2X>i9!`g;?bMc~IJt~6!D!NbF2{8nJ}^i(-ZLd}P7W^XjgC}s5+kT-pE@9Rt44PoNUz?$>;#{bZtL;dzr&?!Ei8=uQ(rx-s&gncx5PPK!K)c4p=ZIF*I1iCall%ix8lNr{N6lIgoi2NueyBhIhpQSFEj>3csOJ+5_!f#q1k^J*rPCrhL6MevgQETA0s-_>YwVj%AnF^?Q~Vg_OpA|l27I`$CmG5Cf>0@p`H_i(M<P(0;{j<7E>J_ZpNwXl2LNVH||Cvl?dSGaD;VVbTZE&B*dxFsTW<rEV->%%EHUu>6mL*t5X^qsP%2Ny-!A<BKcRJubsilN=5pU=#=8>VA3zdao(KHl*@t@*F{lWHRBERwB%x(MV|rtj(0F<?2G$#$K;}B+3E0$0YzOFZf?%8)t62e$9&;=)T1_yX4Z~eflzh<Qn{d{dHCxFLfTBFlk%ov!voRWVdVH)Q`N#A8+VJXs=)WzCdN%((^Oy8JNXV34f~D!!r6l{1cpHE*~mJZnu=so)15M}Gd)vgC97TaoGpKI8d->{ns@j1vWVEEuPfwX#EX5%7VewXo^J?}&lT!ClY=V0*s2v@L*n3EADDp320M<TtzW&a-Dh@KX-?UTWL4{HC#}KVZO8JCjnq4wmASZuN=3s7(N;@HHz<(lk@r1iUNPEd9mYYI!^qmr=X?{3)Rj3s%0aUbw4a$~2FqZ|Wfhl)Y_K?Kz;$f0H%MFUOWpw`%xRnAYR}nY?D|GD+twDyt}b1h%mjq{>tz+1u@Muw41PO4X?hn$QMSn(8s0?a4Y3InyO*s9BoBUW9LpF^95<UljqwbG0C^^M=_&ZI2B;@cbzQgejxy|`5&az6blP6()B3(Sfm&8iU0U3|ULln3bBrD_)L)b`=XcMBHLcQd#UrNP#z5W>S5B*Rywlr!_r1A}W>^oq-K06P0oKR<Dv#RMEgx3c*p2SBXC~LKXM4w1RJ<_Uw4JGzaR|cRJ@t7{<-}TFpWSU@+K;}6m+<y;2k-BDbd>4t-^DIh-_es_Z5*Kwxq!KoSj~OeGd;;Ch54xa(uR(ftBsCivC`eqIcden@fv*kwf)_)DFNzJC3(^gn}b=MReJ$kGlmrc87H|fe%;G=@ugHl0%nUvltn<)aD5sVu}i^)(u{kW?($aQl7))f>WeR(dqUwb0V7}++}r#8`6)Ig1#y>QS9fOEVoi59g{K=n+9|5B%ta2iJlk;&0xI|Yjg2!6t};DHJGfq`kNUyNNulL?J=ZgMPEC+T(Jo~3=%B1En@zKMo7$2sl==JktgJWAsJD%>M4Nt0^K1I-E%B?urHL^}P<maQHI})OsOFbb>s0O)C7X*tuVvpJK-HD=+QykQ<a2h2pFnq;do-6jM0w?j-bix^tnFgE)sRRI8ig%yL6-EJTNFrdVW$2%-8LD|kq?KGuVtWd>*!9Qa%Yzr$dy?aYxB*P6%=~RVZ3urrCyMsDpnQCcNo~Ton2Gf9@{K5hZI+A(osk0r7{xqYp0#4Dd23n<SkUIkR<x7EilZ#w0nfw*t8iCs5YFO-pMdZp}Q66yA8m)&@38hP3aKD(PaqQWZ99qQ-KR!waxB0n;@6w3sUONn;pN0>)Kk^@(j(iTdh}NeoMJGH+ckJ;5=Hr)I)@{+U<AA)h0D*#M-@f)5|t@fwKK)kh#R0H({5Dou+igI)0S@|0y`rrenb{4F8q}O)<@oY$GZvEmBIM54IMhq)lb}?eFt`HHL9_UFUfmjC5<IZ(kj5Wc^9nv56o1uV?Q`R_?#KxbUJo#ZMRfLmpADx7j9H#vz)^xKSqcpKc_LA9y}TRt{bXwAbnQH?G=Wi0e%ck>;RYn`A#@!_N+nwA!K;qmm07X94!6%%gaJ8V_(B?ISRJOxCJ#j<9uI2^`9Nd&%P&&iMUvxU!2_u-0n?9EgHP$I~Sp>3=|S?jU`2D+g1Zg!m(=%GCK@QlR#w_a}DF@Au*3zUS`)dSw25d-f=cUA&|*0Qhy_+xYSWF<&|sO3kLh#;G%)YyEAFNu_f`A1v>OS95V3B-q_Lo}c@7ySU4v-x2H{o+i;`t&>A{D#UxuPXL8ZG92%Z*iekOjjyy_O3iA=`1bifv{=M5r?TmH<!M4sxkkl5nG|62Rs8kW5J;3iGf$^=%B`lIyL&(3!^{kOw!U#&qIp*D@NEBX;M;O5*8VoBy5NR1ia{?_w;%h94_Yxt9s@3J|1IuNc>AKAV;vXk)mv0<!;vN}1`69yxaUI5{{QFdHJXXlKMq}dNj5AJ%Lk)xKCd)Y^R}n@^^NS~Pjxs$1AwocrQ@<;`DnIiJi~^1g4R2mvdi6ey!JQNs$QoedC8n%Pe*_d)@<KN7w4~CY`Wx8FO~$n81LUtv4?!q^T{3C&z?9EcFG~5;IOsPxD7KS)0#n7d%YS9eNcQ}n{1eC$K36;PHxiiB8#wq?~*;Cs&!()gu?b{V7jknljtA3+jHVMUu{?2?t}?qE&tT-;R6_#R2PMR&^n=K44(j2bVgoMyAO6g`AjyaO1+DphvE@)@5a8W{Z4)GY~8W>wBgYMtq+RcxihlgvsZspYBvG6jatEQ4%Cm^6Wtt2@ON%IjzXrvz-|r0D0-cu@p#5`u-a<r;5l{H8KH4kd+uf=n17i^_b{u4zQV<~%3rRcs1aFTKW5)^TMahz{JzH_yVy@FEc(7Z?me{k$dhlQ1e?qGu6g~||BNXks~tx00b1A=J7{a@3w5mppCcDZucX@M=FT)P8(<3=MBUS3O4I8t#7p<Z6s2|DnF$Ki5yfeJdOJLtWvM^LSaV%(1$Vp7`tQsAG5eijNQ|*dTNkp!Zu(Ki?PGRSC2-$nIOzlheeu&671VC%-NsIrK#wL+E=Jd#d@?FF0NOcPe(!KkW1oxZRJ+fDazM`HvCFR)Z+(+Joz{$pU-@a|oEw)5M!s7`lS$?tZGzKrvHkUNh1<Q4PSAY1vPA81JVDmuhTuQnt<6g)KJ_I%awf&m2e-fwZ*g<+kO<eu?K3^Lqbk5n)+UcXSBGiU{kG>HZT@$XxuHIP@s}aY5UR#1;$Y?R%H5xo`>MQs&4JSlaS!B0?!V0SVKBlI4}Dte5jfek`moVz$79HEw*XlXN~E*t?q`So?e&VF>(4${LhV(={vXuR*HSr!+l^tWU7zln?*ImKJEi@!f#`;*)MN#omh}v8Kv6yxm#KOkAj1hWT!a!E><iB3K&CjPV>`Tb<39J)3qalVCeAKDtd&ZGY_e;z2ry*!u+VyxBX0#M<@wp;ypXvst5xddO=n&6__%b}9$#hS<A6^z>1s~n&R?S5QS^L{*>Q)eEUE8)>?DtGXR{_62g?CHaD%px5b-D1h2grW^|&WSbDfKBgTfcPc;{4cP+PHGGxi6w(tBZ3rDs*PuO>jPM#HElt%_$P4%6K}sFRiVg2MSH_dE)d;kY&bHZw<F?JmW;c><S!c+zj9Z9BXBDQD!{_jv5DtKi-GIFPuJ&4#58Zs&$8CisDFBP!WW!Td5DUYtAxwT;W|U|aL%;GBPtpdLQhdktvn@%?ST-IxMcOL?u}axrDvEk3Rtz_UFh0>o}x(i`ut*z4dG;?s6~mhYTLOH-iV7pn}IF^6r}V{2cziCC5SVM;E?!&YlJZG?Xhu<%zs?bfb=zr)8)7NORqXir9)1^(SnkW$AtWn;pP!rD@zR=w`;cr3TMl60;)S05jzK4)3K_^ryb`&c+Pr;Va5Cw=3^S58Q++7AQaJB{faUH6H}uIeqTZaPJSBr@~&5lefa(%R3I^oe&?+vcZ-Z~9!w{PgA%Pl~%l-i@M+&13TJ_s2tRyXu!Mt87<W&KgJ}=?!d1RGhr20OacF5PP)tScsj&#%(y;OLl8aDJh(`SMu{ch#$>!9m}@eVjMB6E5}JAcG($^xwU_|y^+b1TE%S{K*hd^wEs^1rp}wI&6B;XrRCtULIR!cW}xF?xpDtfGEw&yRkzsfT&xp*8#RN3pLDvXi7E}_RdsAuCr(A%PSxr@1tHj(4q+VoQ_wU1@{j2DxiY@Pab9_5X*u{A@IjS}BA;(!?rm}sCDw-vch4KO%sv#B)xIl*J_wTAQrS4v+wZxmf9U%<P#bSxQo3$w*GwahTh>Oip*U#=kF`2ruJfT-Tw&g05AkP`UtY{MEHinU-+J@PD`%Cfh9PjcV0NTeGo8t3x+gum{%lp6A;KQ+&t`q&v@do5<8#k3gL8i>0k$GPRJ+UcrOxHoeZ09(W9D1pIimX|%@bia*fw<_c-`#wo8N_xAu2XRd&}=?ucdY>W0ZjJSR=xxO{%rXLb9>0Qa^}YS1xaanIN!DEfZjR)LzDsAN#!>q0(Z%EOISLlXFBuZkuyIbAE<b6(%%L?Lc%v;Q7N{VfnnRYkIT$ZeGmF=eoc&INH5OY|g^<M|MdNO0w-nq9aF~TsZq1+Z=`F##`v!4tGqmR%vD%sV#>1fXWAbV?Vv_u}_&N&bhzPh!t&|D8BoH#O|<K2ZmNXE*`YQt=0Nu@F1_z$dR1Zypl!X4C+~yoPO8U@eUS5vT}aN<}1D6fn)tVMk+XBdOWP0Xj94dr-tHML}lva=7r^%ZC-J>Hz*&zeR9@ct_ex&Sl3TyxN~c_^}46kxaYfC@fO$3`?$JCl;(}iVLnb^t>0xa2&)M}9pA9#-ouX1_IBr%)qvNMPAG0~maC&z=!XT%aTl0W(w7p)=F{x-j1=Fbl7<2Yq+1#rG(ogr+TKls_~?#5wFNL4pC>>u9=69{5BfqilU+^zS_pF0T9?We#t_}9HE;0>7B71C)q|5owoy7n2i}$%T{rYuvtoAU??<dXSqH;vk7^GZl3n*iHQJ!fA7dhQz(p7u-`5>z``ktFrdZDW3U<pYF7svwPFY^2iwRq`9D_ckkPK`q>98ro=Lepdz;VhBVCSO<73B}UOjiigcT5f+{IFq^qH!6dlvk36*Z0}#UvuGV4o}>GOhMv{6}$HW5Qq5sh2C=Xnxi(0cBaMFWcLKSC?BmX?W1f}c#Wlf9-fuIt7^?MKUB}Tsn}o@;<nFW<I#1WF8AAyWZ}Zn(yy|t^sxHu9uWO*xfSps9QV+#)6?iLmj0N54P7~TiD_*o(I@TAx8jq?e4Q;Fk$9r-nfbCtMFm)~cXHVL1OCq)c-@<gmlvPM&wcH;YiLfdK&H|NdO5)d*RVy*QAN+oxy${xEt*=w_4@(^tbnm{RtMsr8-n;SukPx&?XYBeXZ8aq6tEUsyUqG`jQE<z-IqZQqK9jrf#4MA+%ISS-5OEI+kNa?`Z@DD%1E{R4z{cQA<IXHs8*N3I@BM$<H+JM>R|W%zADXy-gR1c@;)!#nclY3*bLgn&g7sSO>}$voqis;nK`+sF!Tvly0wAf)VS=b_da|R4q>)}9a);xg!Z7PjlU@dsXJ#yP5B+%@rb;2yLSHsYGM4PPm%T(qhXlxQ-QvIJE)@LQ}}bSTd@7eXAkLk=nsbnQXPYV`8+&)WUMlsBOEohQKuRM4@Nn)LQ9#W8C$-~=Ik|XLUju42Qy?eIM&-49*u(!0$po;)X4^2P8Rs^W1)@c$^OQP_-b=uqF$Khc$RcMT$+(J`nao9?4esbT&ij%l*|?zSI!N&trp`oH#p9b%1vopu5V1mD(eWo+++T^xtUN4`rQWOli1d;)xNABmh!0IuqXO(+9G!Nkkgm+#ju<=bS}MGwZk>qXl3U|aJBU!iasxgYn<uuyVdIIyr{0uP=jf@8@)r=X7k5y$f>E*8UWCiyq>TJh9yX%^=~;1H`qHQ&!@pdk-GYkZZ+~U8Ok2W|4n#%H8Xt4@YPnf?b-5{(T4O2Pn95`y{)sdufWG>!MpNCt}q~D2oIcG^{|fLp5>-8J`h8+Qk%*@IUky!$uEDmE|^Tq)?)D{Va?5i@}ypdKL4TU0lU;fvC;VP^#KXd%*~x8l4U%_%NiKo`tPh)SzeF+xh{bR_R7QAK#C46%Ich@Wn5a94}I<c@1WiuO`ds>)~!fYcc*RoL=0rZ-e;>w-t=~2yK39>)-1$U#D{fT4bWTzzq5f)yEt|i^gDAAwrY6pRfoa>46Dd5Ok3bm=s!O5#%kV5mX%?vZvJi^8*hx2xx#XnPn$c-dv^7%+%WScM>sq_7Te<<>Bjjw_eB+~-WvN#!(N2XVtXk%bpyS1aw7o2Z}(_jvC*A>?D6`>p511%Ubo^_U*NA*-k**j9G<g^ugtdsaja777dCm$*f%H|+Lti_0GVAj9a)&wd#*rkkEbSOPKWj4y7{cJjoJ0uXEi@}A06uttpA|&X1^YtNA`Y%&yt#BLA~>EPQG&%T^?4)y$iUR7*_x4ZtyMI{uYux>sH;(AqmyM58xi&1wB(19+ouG&S@0Yqr-6pWWdBdJ$9vL;LlpoNn6x%f;2&|)p<O|<PEQ2-n8?DScNbu_-2kufTVq#@A9Z^jCOXL?~c3@gWf@5iCbnK_}8Ol`y@vsv$rYy;b$Y51lDX(C#)LTol<zO+yXb@5~sNL7)$HR@p}wfxJx~-WDi-{YpHVR1-BO7)@DYJDN;KS%1LcBO#-EM1G7~-Jl`L^YwQA}YXGF^z4KmhNW0-~;y2XBPHz1Q3c7g9T8Gc;bCCyqEgXGM19-ijpm3dcovpFduJNc0$Q5^wD@5dwo@5}w3a8c9hFuPa?c`^WtwCuocjQYO9rvx%VH}JqK?dF-XXQ_-Z=pBM2jJ6Xs@VHoJvv;Y!Eu%76u$<Zp|de92Qc8^8@xC8vA@S5z|)E^f!s5^-QHQABc&C_+dDLGWbKS255I&t<}WfsH0DwDM~!-v_NalS&mqy82_pE2|A?7yue#q8J^_D6>ybz|=0NP&wVz^E$c?sA{W|2yw*O@VSkQw`(77QyZqu$*AJGJv8~txI#r{~M>|B6T^Ck#=u$fXM>ZaQ>UeJcwV^*TKen5~+xBP_%Z$-%QSa}Ym>{^Cve{(hon~`(e9I+<bn{Vp#nVXeD^qset0{NEu;(c!nnAdb+#kQ;wz3pXmI)Ogx9Z7BSYxh0%dM#EW^lg2bGsUGr+2q=zI>?y!EN-v>Cw<&yb)6InfI-evSxI;eCzbhNwRv>MYh&=1mcXE_lXSYoy;0&eb6m}7eUq>!YN+z=b+A7yM%ezwt8`GsPUxO(C)la9?moI*ezw_ddAyD?#X1iGx1C&LKPz{Pd|o9q)%sF-deg4rM@^BHZJ?O2D58e<@TM$-UUBJorh2Yk9_~{%?Ly$O#O08r&dpPHJn6`*86h34HC}q0qb={*DbJn(WTmBKh53W2P0EpAe<~#pT`+ljZ>@Um`m9WEu-kUkm*^~u+BF2m7MT+Jq|&NwdGxiGYlFHb**HdVTuZ6Lx9%$~qjaatbNrT~(rv~&*9Nnqq^G$GZr&p?<hA?g^<9qzViow{yYc;ZSI5%f*_`*a!%p$q9$uXVBM6U!zOu^1jjk=9Hu)vg{$t%~PIG@Jh+Jj3-Bk{6Zm9LLMHAKaFVs>Q;xe|6%5Y2OqGyZ8XO65l*&)0&XejJ#76{o`sBP+Jz51J#!RzgJIM0o@8mOk4+!C3R6*T*<-cGQ)rIpdK;m8YFC8UK<M=W@-YgnzG*v`0<oPk1Hwx0*V?>O2WN1Al_f*grUaBhmhr>vU9@~bygkGm~EsQZW;rPQF9ytgxy>Jx_pu@vT;ANypE5?(NUfAkkJ^q&npl}tH)=96&`o&HiV1F>#xu=<AMrc`An<`7Xm{>s*d(=mT~v=-rMZE#V&=VTrG`eX7ym@!E9Z7$-v;hYSN%m;SUD>o*6oc0j8a+fWd3T{B}!YgALL8uqvKHf2es((ws$5W*yUb-{5@l$qU*E%H}e?4duo_9>b8T#`Y-wt<zz~687A_|U{n>NRG2A~FK$zeHTyD~QhX}V)W>=S0qjW$^npW7&uVr6`&eA~|>wuMP#YM>pRqlSa+cG7K=!i1(2gMX~%ic-yA(xTlJBJq-~JI|^N8#8o1P;kK*J1NpjHY02%k)nS%%<xb0ULMy*-FI)!j?)p$pmsxvuZ(8YP*I<Z+<%>q6ZS(s|5}N+hr;~(rG%lQmroGgA9tzUQv+gP-$bS1_m99BeqF?u-=cS&{^QStCdw2|SN5O%?Bm7$9yeN>0a>4ah&I~$CLR>TR(-qFpBg^0(^_c?^ox!}x)s_`N?7N)lDGQk^)i{x5m|Y&``#MxYrJ?NzxMhwW+dy%PNFSlW=xFtcMBu1w^e|qkIUHGc)z3mkb!J@U!MQe9800aWU*WfHWsHAeefAf$sTRNl_>8s2{JDi-BsZ#puWeLy1gRmr+Te5)y?f+4{X6rkBJE4e0P(@;-KBO50w3aWAj*Xn{BkF5~eunRjbpEKG^2$xXi`Bl?>*bf>CR?J+*N$BrJmSsobqzn(IOR&yB1#3t`VMs90V5qzwDSd6HS1<)#YX-^udFk=_y8jIQk=XdYO5^D@eg(9#%*fXEN9=@u^Xx(>gyY{G}iz?N?gKb-dFdLy&ezc<+N>Pr7g^apixmU6B<>Ep4KAw=^)%qbrnR)R5d4b%M|heNs3atr&mx=aT|X6(Qd8@@H@UTq6F+<ABbNsP+XEiss2ww}&ma+o{^AGtu@HMtGbtFDX`WK1;7hi4GA+1u`9@Om|3G`}?FM^)f>f?xw_w7vSvH8&KAN%k_|R(~)I=4R=eSf}SUbf$gBUJHxwdj;`RGm7a7h)30Mm&6S|VWl*=0_4dX&rOpX8)a?R@kXLGht%R7UzztH=v@?ZW&P>);igx=m~@)l(9OYi^g}vE$i@0Br4GtV{F>M+7{|^Ih5q-hex2OUshUh=FFrVb_oMAEz+U3STfhj5A&acnu^@M}oUxbSid>~KZa}Z+r#?ku0@-zqTp&e3_OJdk-&t>N1OAfnm#E}j0v|_eRUxL}6L_DWM|~*h{MAY6Qx<R2O1bwUdp)+@X$H{Z{=Pj_)+bM|hqlsgse}34p7-PEKE}1_@=un}o1F`dz18q#x2i^O?zigad$0RFSg%I;I&}cTl!mo@!%XM=_Fk}<fFQx_LdY{mCZfLR<tn`3`NcM$V&8o90nbh2x1sd3Q6t-mDXp96`C0?K3d5tl)%?4QnT7Bi?dr#{^2?81vB0+s1!O}l`fl%w9tE8j&%xUt@i|~QJr>ymZHphzz;ahw7*RkXdlTJ#7w;GM%x=f9y55se;T@*2gE{0CX$jN-zRt#_p&Q9B{{|u5E>Zq60G&s}6_E*&e{-1hJOyt)FuR>p4X1MX=%5QS(+A)`ul%K~EF!#uvOQ1OyoS?5JVLtp^j>T4!etFk(?>jO7s)`y^`H)n_O+E1>N~NJ&H9?S(^+fmOpNrL4eFckZM1}?dp%!=6BV3YLgd{zFemg;zqK#GBAa$GA<&NR09kDY5B539j!hpEKqh)3+VTUxH%CM?>yd<3L@sZw-?*{RnL4M$jQvQ)E{(&dRk(=ikOd9>-(n6r<JEjk=_aR*7h7Y8()fbILGjlmUG1f*uczQ`A8<+{KZqX9pHR&k31WzENB;XN6Y1v-*~jU#5`^7fc2G&AA$Nu+)yJ7X9E(}eyOuWC>DK^k4E3SKVK~4p)IbQE&o-_+d$;8%XVn|&?)aN@dZ`san@=mPBkM+k{A7h`SMk!|(~*9xXgu1Jo0ZmzXCk4i(2`OON0gVqvd4~jFk__SKa&C%PFF4yp}Hv}uS!QfbyAxnM$2PGBu<;ZHG8WLIeY<n6xPA_WCJzUpXwPUUQpPW*R1aGnbnzm=Ef^Y?o)V2*Vm8xGkpd|`Pe+Inb^(Y1J$gP-|cvme=1m|npQV)C;?a|iy#~|k#z?FzLQn)Kz-n-4#0xndf%F0#+^}(XpD~2aQi(v`)ybNz2o*GKe_z_VWoK|-+`2bR<+IIlv4AmtV`pO%XEWAFLAz=E}Ww$+=f@yCkWe&3<G~*+`Agxsox}*^RC|N#kJwY8tWTL-j?R)DtDzfNB2w83!anRq9awUdAUU-?lb-nquZ*K!szUe4vhmY%=??EpG?+C?YSEiI~vo)b9K41n&-{Qw)?b@+pUHRpwxa88;jNC`bFaV8&j-HzOGGBXouB1V?h6lJ+R+>ZSqxj*=h*dA|YmNCNGEPr$7z_c=q7ysZj01ZT()3?5qnDkHd1WQ(|vuEZ6Nac`3R4xz2c`Wu3iB1-a5-e3}4{x>qEo_L`^@5qXZ!pZox*HgsezDEF7lGsY&SPXs93t}#AWm<ME-``<;l{`JPS4%IN=z3&WOv^S}IOdL!hY#`q38Qqhn(A{ax?B@RH51B%LcGn>leb6GSTK78BOB>S<y4ku>gkRfTeY$E;V?hNrZVqW~+ftV2=JtGa7{9;UJ<&rYr}CbJ)}NI+K3vTQi_H+gxORpw?fInP&U@+g+UYu>=6{mNs38ye1AJRuYnY{{mbaYmh?jI(IL(7w{AlpKrh5~e1<1*@R3QEk9ec;LcOb-2>2u_0D7A$#Lcf3JC=2`@p~a)(mA3lu(_IUD#aT7g8~5<-SFPd`G>%{RIh3etk^iVf=Q-G(xBiBfBVfCkyp}RB_un{I?wt;N@S`?5dp$C!n%!(Pe@(Wt{_FBbC*Rgf8w}%4{Sf$_RyBsfBuFS#Tc!)mS+C^K8nBF;;3m{{tiOZ1oN?=Pm)08$Wg>Jl?S7AxWOmb!0}ns^&q*IYx_9eeY7pZbc<nrNEpiUU_4{6~bg8Khc)E7MMh@U!wchWVeBN%jupZBNzt=Bqe(Tm}E21F!v$bzvqc$>VT_!EO_6?BY1J_S2JqRN6xd4Wae3!CPcJaMgV70Y~qlbLBb0b|Jp{6uiUxS)+sn{5Fe=9cqwre$Kw3fSz!x(>+z5ch~Q0lw!gvdsX;X2Q6T0Z<l*Q7EVX;gb4)0EY({OWiW#7T4k+Km2t7U$5}GxyKTUQB4u;NjNzE#}`x9#Pj%Np7~9wcL{FM%*DXSiRuN>)!8;sM*OVRj~@&aTG$MSIxYnM__5s3hR%km+QS!^VZ!(M7|_XeR1aXp>%(4W7h$}U(Z^B8}6nzKT$mczuV-@J;ul!Wm~;*WzMJXh5=L_N2P`1@ROgqmtO5VM4=}dS5El}tIWp-w6;&m9UGHn{4t5|@N>V6C`4zP8OYxbYeC{0B{u1?zjt1kQj|hRoPMcF+jH}~-R(Q>Zr!s!f+1Ot%I2m1T0^eWwa{WfW4-<$B_-~$Je{BFwME<?H!2U=>|tFN8%ua$24e`qZ?GeB(|7sy_bMOETW4IX&J|<5peobP#C$I&8~qcaI*qHB;Dst*d>&wXQxC}q>68AKdDD}?X)h7eseQp3U~b*P-qloml)CrKE31cdT(1odb|6%5cSrDycF6UguW8e2`j?_Qw4qkBx#2HUZaVJO8u=mG6J%?7M_li2z_{cPa<ui8)xDaxUHHb@v_lC$pyV59TfTlofBKp(T-`EZ5a`Fnu4t*98nPKtt=(7?i+0Qf-8zo={f+L`vXsCXwHO#*cyd81Y&5W9?*P}+M^o@b>+qE_9p{(Dwz~al7|8rJuG_Mm>%col;Q<?LL7~ktnZUFhJacQ8X9|ys>3w*L-Zdxv<c)}>hMSR9pBU$UR4e;!li$)Q!#4Tb_wg#&SQ9urpj(F@%{?Wm-oD;klB-g0rP}vMYzjo9{jjRXbJF)lB`x^*&Gz-d!+A9g;W;$M46uYw>w!ZyR(q^M#^dEWn9c-x!2sY?0@k~0mCe2Qta3oRF~*5KWe?t$8n_C9Qk&ff+uJr0^L{;}F{t4)vx89d-x0~?#DA7--#P(IgQj2K;Z$3j^88R97Q<xDwq#5i(D?vx_3yh<$vtR&KfyZtQ^xpC^(;n?vr26;OOWtBpB2xIj7G}y;%y20EF*L*W0%*#pEEr0eRsVH0_NV1*)J;S=wT$v6G237ivr;g!(X38?{k>IKS0Um--tWU+ws1W1&8e@ZfrlJ`*ZufJW9ge`jJ3-@IvuF@o#red>^<rbpBQDkZURL97{W1quZgka{BRQ^uZv%bMQN}<J8I&r8x-n#$E6q*fIV?JA5ozoR;pK?Y8mw&TRV1UUzG6H=U~8`naMCX_FZPq@j*)TcQ-zWW4Gq5f1?V;=@-8aA|$d6RD1$I%-v-N~XiF&}uQM_ZD}OQTC0R_<<#4F@q_td#}}M--&=H!<Us@yFNNY`qE+hnYMZjDoeY&(WCmaEcpDDfFnUZKBo9$^6Rf*61@}lcuT-cU?_sBK_nl5J$1|3W8_HIS8#dRjgu2qHh)JL{7sb945ANN+U*i44R|znwf5gfK$vDX-ES}YEu&!*?~SUzmcr3Quf0X*7;*(r>&8%aZR_l_97>nhNAhC%k-L)a(Wp{CCi}oTht<2>9JQ($>hcRfpAIJSwx^t~Lr&Nd$_sLG=RJnfbFkjLiN9W|s@BjXS*McOm7?A5PaSy1X?(qrlF4qb9UV4@ul0BFn4Iw2=W8ZwqcWhCa*~0_eFg%~)#Rb`Lj<SVC{sWw*H=^COtn5zNN|dV42gS}YF)+63lJ=q^;`{K4cSH-Hxo)WT<6ocht79E=39>#c4!^=B49u)qQ*?9fG6F41gB@QRrT?2YBsM-h(39Qr2uu8?78!|YuP_nTd-DSBa__x)`;!>(qUp&p3Um6zP+@w?|pcr=IuwdQCJn+aFU_V>^zI8jH~Bq+xS2o@?Leq;hlb8yvu{GudhjKqS%Ji#OPkU9BM|6-=(3qh_3p(*lk7rNlklN1@miuclE5wJrmx))EA}hgk6gj!&z}V9b3Xm)JK59*;gd+pXzdV*J_i~;f+|)b=$^?^3ytr=OW=B!ly(5w$Q7{>G>?Ce6YjAurY{FjcD3vCWm2)>VoYYFPHlE&2L^MHA|TL-B$+J8T}}pprMyU&HAxnP3#<0V5T~sbB|iziy~TNgKNCqC5!@U?{k8@6>~l7?5i_laGOP@lR5J~`Kn48t?6s0!LpZ@+-%-ug3f-g*G6-OwO>$~IVPL-@>cR&E%&|FkNurXA?+S$>5$x@vTT(AIL3~xgDq4#dG&hrS|+#D{dZ-+<4$z&soNPM!;1%F6UHIh{4WT<6TcV<RUE&b^t~W#KEu_zHyV*c%Lny+Pwo(}BgwWaVfT7nk&1h5cuXR+!zcW00Kn&SXj<5e0PK){J)+fnXfyFO7|PvfGz;p(&1qx8)-9^)`(1h>h$UKG+vG~scdZu@y!&2utm~u7$-fHTHIK*B<D`FQ@y&`uRxv;v&qg+9&WzZC>M0Lwj&D0^&>N&p02H02vVUUQmR|;yzw6TN+IQf1+wkQbY$zA;_<qJkNCisc?WrO02!{EvG1(C7VnB{@sgdB@^_$!YNG`E;zjQYv13+4}dv~>+<Sh4j+^3CBb6#CccKyw9-p-G<_*r~CBq0_~uhm%g=lmXm=jphr3Co!RpXq?8bj0T6UTH(M>Ph{^MWGL9LZutzs*j&LY`b=fpC`3{Qd7H!E;~mx8k<Yzd&KtIFUPy-p(O#|(FOklygg9d*P<^VpRy^h8XIo2+7_+Fj(4QtxNh&~`%6A9K4~TYEa_f;s>7$z3PJj-Wu=LRy*4Yi&kN_YlV54+&KRdVYI%w`vtD>v=1xo;SFYbY36pOTpdyh6G(t>w^w%RJlvcYJbP%cJ?~@$n?ysSiY?q^rLI4lQb$yKB`$n45ca1vSPbD?#67l^0ecu`SN>frX7*y63pQbxl&vC4*Q#GDUYj7A*pFsS)FTa&dt((s2MqJ>PdD-*G?g05Jr*|NpoYyOQm-JeagzWFPne=%P8Pr{9!0Yx^#Z^d4t6Hzi6@+vfJ=}_BT~=+#C!CLPPTTd#>rf_YcMJ8l0SgOq8Xl^tcKAA#v5eQSFx@RLpL7NWquON74ZFxk`>BIfx0QkfXria|@Bq$no9JPQaL#kSxJ<{uSA3_}6r+fj{DrPg|3ygH<H!&Y3^iAcn%24-pDUd>@?y0TCLZ)qZE|zoAD-NLt=CWUk8**$sDaqIa=RQ75UTIO$!p}!wM)XT+fp`BHhC2vM<28OLv;~b#xo{IIy$8kf2Pb!{z=@nXr;QVxK$=t3hO}1V*v*`^=QctQz=0|sK?w0FKUhCk@O}e{bJA}fSR<5JDN2nm5wAe{^o%Ky!Tq%BtJ{3Wd!b2mn2x+Pb<0H;<f6o-INd32cK`iMY7Lo&_hZqtFx4bEw^@mwF<ADwJIZi7hv&bXUn*(H!%QCo~Hl*OAuUr_qN*C%x)>Ib@Ek#dpfBPCFUv#6aqQ=^AeyQ>DcEvPWbA2^GN_~rd0Rmx%*vj&KP3&qM22qHE{Lxd^}LoN4FY-J@WhNEBnH%PxaVkK3qb@TKg*Zp)USHHnQ$+!S5j(e=0@moE%*?v)NBSpVWpm7pT%Z8rO6W!q*jOelR-TF8#{#p{(A?Nfszi4U?YJExJCGkF=6$qjTPFSnmV)hrKxZs~`Mvxfw2q@ARqt#`|N=g2Wm>?)J%}y4{7p6J4u)df~CjH>1_|NFCphkoGnh1mokx86T+HPJbh;^KjE~z<e(3a?@uKxIh`XVCy3}{H)(jcjzc1`?7pLIveiH!hf`}s*Jq(JeEwicP^}Po4dD$?Unr)k2(om#tFI>m&_~4<HPi?s1~Oe(Q%xO0#D`=-59&(BrDLxV;xexz17tw`A>A$_=vrn&+MvvmzmcYh!EKV=2bV@*YgN5yP%ppqvS%Hc~K^Fzqdahf+?SR4U#>~)YbWFb)WMeOgo|ICQETbF~fJ>d^(r;15_s$*|<D$o*^b(hYNSBxo1w6`-(rl^@$&L^)b4S-@B4yR46Ep&ku_*>raRXw7xMhnqE}tLj7+>Ed{!d8t}J`soS5>lA{9ZKUZSBDZDATe;S?wj*EEndh0`BPcFIu+{rJCPm!My`@uzpIt8FX*f1q*IcJ};zL8&@?Q?iItY~TS3Y(I*>Xf8bYD%<&+q8CG7%jfpF{06Be@qEJyK~yCUq#G@j$7H0m^+8l*L<=<Wjv+*ViNv^p&Og-+qw~ARKFUl-)Q<Co~M<INREV)glkuEz07n)As93kljDaqZm)FrsVfaJ`82`z$JV9|%No`BQ;%gg(FXAE)^3>~w%JsOHT5TDOm#|f>Y_P74)66?;95%;uTl2W_3!*<uRo+sPx=f37VGoiLxN~|><x3f4Fy|~XsB!zg3+s7hr6Td(jfTixNmhWtcG271Qn}%+VDI~dLg)SWZNg`G_jkNbisSvs}T}U-(HElhBfoK;#5;?hdTfp9HXvtM!Q<=aGE_8qes5JA5?sk_-leqOV_^Ck6Iw;^#*vk)dJ-A$Epa`fy9x|z|HNG8#l$q>R8;bR1Z6~4JXLRPrGizEqK0pet0t6E*MbFKY#USUKAcxqiA^RH7^w=CA{hFq;Vd&2nN#kz~FT`u0n!vxZ*;^0)iL}QS-0uHf13>%#?xS;k-)}O)Q+OdlHEISnJswT%XBr604VC=s95$8(qtNp~9zEds|XWdDDdX{=L+rb1WU~T5RMAH=WD;e*gBH6o0)OA7QZ(AOZC*lOYELYBi(6pfUbf-Kv8q9>z%7wEJp>wXu0$m8N7$1owK$8tsf_sXBdvs~WN_n@?E<gZ8R>f@hye^OB6G4Xha!hHKp}jUiB?^UEk|uu36MU;w_JtFRHzCVfY5qZ^vN8*bH-b;rOa*qOO6ukP}VpBt_wm4LXBm#pT<U^SYv1K>c6YISYD=p7o3&MqbtekAdBuce`tHU2q~1opg<Bc&%`pz9LPvQXBVRNpxzDrWMdn;h=@tb_clJv`wTU{yayHI0|6M_ve!!=I>g$hOu54nKZK)RILs+Yj0{iQuNRZP^C3TrL2ueLCqX?Yvi9N{rt9$_`kalxIPVdQkqo{*5n{th#%$muiuPoB|@{tntzvIo}EuE^tR^3Go>J<PhS)?fI>kcfZ%R)t!95pr)3V=%1$`y1~1>5$H@tAfkx3xYJrR8$kAGxLu=~jPY_9^`-A1>DIUMpf@Gf<Fz1EmkayC+_{|j@H|=b7U9d>V(Vb2%oF1p7Io$$PKKwo(_9Jz50V$(%<M|DKA+)pXg-SCTt|7VW-b14SWBv<8apxi|7+pR9*&R=^vAH?I}(;PMH^}7=+@oRe+jfwR+kHxtdoh{>Sh%(sG54mKis-)mS5KT;{|!K$WAV{DN*RokIyV0;mvN?kvwY%w&PpSzHho!;ILOSIlTS2SnJl-^#Ra>J&%WSySJ=Y&VTEzcE?uHURa;r;)|MB@0O?h86bv~aJ>m~T=$8cH!aIcY=^%ZyrZDzAJ|zS)t~jd?H7DGjR?rt0~bQQ|G})#*<gNrxgP}enm2_kI35i(`*;sczLrh@bNT+s<H}g|&gtPD_C%ydUft{MoZfhsx`DG&kAHLYn%-4W$L=fV_^mZae7J48HrA7Y0efHB-(xwbt&tE#9$+GszS`TX2T&B@ZGfI<{3R$C^V3b!&OQSk?cRFXctGFpU>I%2xi|f6yRBfT-`clJ0vfb=25OeVpUZ<47E)FuJAASn4qwYW>F?p+LxF0C&xv2xPA7SBWFmU#r%469jpc7O46!K=-uOEb$ZdGaj^f!pf0;>rl1mH91G8$DNMUyuevo@(_P#0doc*J@CtW|Df8##)vJO3sa)>>7DC_x_J7aK@AnN1WUTF1os5M^m^S|+k>L_WPXkY$o4msy)@UM(<bZt=V=)A2qr<?d`sjj$rc3IbZYA2)IRQ~!f(;78(+-e%8TYHEz&Ys_o2h^m|-t0+MX|lR^3UcOvT|MTsDf&KIU1#_XDC2mj-5}o`&+h%z>2vmA<=mgnl$Fvzrmqg!oAtjo0|P~+#tp3ZpZ+HSFvykkc8t~{72JYzv>hRJgo7^5MYX=)&&93<RWO?UrjAqBU;P_Pa<krQn4g-_W8&Gj=cVl(BTif5wXwv1q3L+HZHwe3Z5?socm97b()VIF4ClA|v2;=B%daNrzBLyHRjg7v^(8s44dB^ECZ5&%jjA_+W;-U(2KseEg=G--W;?QFG?2MR7n@r)c>tGZ{kGG;3Ygw~|1xq$b;SyML^zSFlI!estDrx>7UCk)Yr;)ujA-T3@)b01F~3=#9p?fa18)E5P!i#;cr_d^8}W&F^{79v8;R>Cesdem%s#GyR|g2`Ly_5rQ3b|YlzZD$(?18_&Y`*MXKjht|GDo{cLD%sn>96Cuf}}mVcD&R<<~z1X|fz@8mZbmuI@L|+H6};0?pp*I<EP)ds;t@ouO8>VZJdI)Kwb|j;9OkGQiurJ^-LOunxQfrVZ!bX>T~wVc~rqonMXPgS+2q?0yNy?Y}Zy@rQR;!u%fA$oy6(8Z6J2SNWie^KB4rCgJ|5Hc<6Z%ggx0&W;e;sX@n4XTIt27bis@9cG~uW9RH&=J9Gje8nfAH_39yylZgD7wQa3{-lt)`Ran{e@6H^FODS6&7w)KP9JK^t-F1iD^dSWB4WZHAjj%@`QwY$$E$Vw%fdb8qbT8&ruHwNdDOo1mUCwJa?x;?OqGES`y7nQ$I05{;vcBfolVoP9PfD}YT&@G)JXIm9cuc{-gUL-`r0BZMA}=?g|?p5$VH|=IdD^Z?LC{V=+?2yDsI~AlE3HSWs%`KD15pH+%rLg_xvRvH1dTp|79_F<qoD|<^9Q<>rk|AR#)owqwf2sHO>n4kbSY?$R3<W)cUyw<PPJXh2%Jj(YgcecfTIj_TTS}+jsDJ%Rcw1sr6|^CTJ7h@X6amaiH$q_)ZZ0PgYhMceq@k-skNE5r)0073*VuI@4Oed3T`~VEBJKn$&b>^y9qclpimO{`XIbo>cv+fLjp@x(W^Hhue!`i1WBj0{FDynB6Bsoxs=;bfM{WG32r8kAw_sB7+f3Y|xGpKKOkMuTH4mIK~!3It^&)4?uy2buIS=d3U??^Xv9di&pzP+G?;mI-hTDeb8Q7QDs}n0}+NWYmZ1R>o7bu+dMw;*4sn&0PBDp?oc<9mhp}goYQ4HEV*;^k}Lao_cZm1jL$1s`*nS;O%42ee*6(Wwlk%M=uRt+qPBu-cPBO6_^!A`{Vs~ps0_Qc3L&&>QB6EPig8_d!S-%!e|C(rIKdY(uaJ!Fc@Ju`=A-xn+BP{}l|M$RfKs@{SPC^7UN?>9$J&Xf5fK)!TQ?>y`(oM)u3OQdIc*2l1Wc>Uw}=ODWo23I8jzf}qt^A(CVrQ}Cp#3WBcMw+n-Yy#MSnuGR}9|`+!`TMa!QD@W5rfiHx|h9K3~7BDiwR18}qqgW7d&eaW^-$U%n)8YS5U$T!pnq;9sD|Z=i9Al2cJI@8wOeHPsnY;VWCoY~AXVsF!E3zZNA~P6~3VsW3nY=nV2d61iYED!Q0V+pfN~{;V^<kKk!tQC#l+=>Iw!(bE*}N7d!al3Rc)n03F2a4tLYNm_#bVp7gtSPx#%hGBaw?GMEYSZDa%lFB`II&&vT$jr{&>TCLu#dRk5UrpI01Mp;`g_I_@YB@BB5g%!5aZY$)EFNTV@;C~M@_gu;(>K9xCkSvmjfWSt{=2|tur}Yyx&6G<%%rCi5AW3<aSm+s=!f6H4RE?|AsXG~5`Kw<cvYQ=L8IBMw2@!G_6BleBE(E(j{~Q_##RuDH&(|IV?f&rYrWMe;f?dVIr10zHrOup`Xv-1X)Igo>1i!k%u6+H;4$tv^Ue)ye(CQG?uYBBb)R7v6U;_$P<ayPW|C5;{_62qf8M7<wDbx@ucs1k04gNE>>=L11{=!E{fgCmGsi+(sKw)4L(M#L{J~@5WbIx^d2eGe-Sr<|Pa#S7()p_c;bDh)zY2{j*UOxK!OP1=Kc98Pjdo`nYI>ka*&o?VgJ9MC6yYf;hub>}@`_cjYN&H+5EDJP`pG~gbHA>OX63U)jm^?V$RngxI-_8-dx&D?WE*$}9(%Aq`Jj;T-e)cU=o#dL@ZJ2SGx#>ykCEn*Y~fv&38?LCMm?ohuL5`T<MImrS*-i#^W;1zaI^XnBm?_f#cf>+FO~YNZWw;om51fiEHdLNNHrPl>+*gt*L^jih1dK={AxADx!<QV;r(+vPcMz$UP#xw*vlm70xh}AdUIj}!hXLa`5#4R+PoYX#o^D=MAI_Ov`9=@B0{z-Q7<Zr&_a@;<+DHk=UuOKyF2Inu4_iOC=hH)#`Zt2J^P9FR7c$eeRi4sS$r66gz~6cd{@ltEtHOvcgj+-S>8TS`Pr84-IfSrOj@mpvM(_#Z-3LX$=_GxmW$&jP?q@P&|}z(e^922&s7tM+XJjC$KB&2&Rjek*r^YD%I_YRr?*l8Pp{L42ju1ALmgjL-I{^*u)bPJAL+WP+rJCRRsMR@=~p1V)x221u3E<$N<oj)0nrdXO3#fK81U%)t>}bg8&K%*Pj+lSkG1)4^J5$RonM^u1;stE`9<DTF1^kNGumBm1u5AVgtzZ+=s$)5(&U{zr?l?T?b*)m9r>bui~`X%sNO0F=(1E>oo68J&P)i^(7P`G1v^pkPqs?J(dPXgOv&Gb=93ZI`jRd_3q~ZS$tkq-ky^7cS{Kv++-G_D5_<r|Iqc`^quEu79yR`*%|7|1-Oft+xRUq#i~BLRKb93M7BI!&TC>~7tF!135&IGb;5dg@VB!l3?xHPRua}40@4{}K!>R?%b~|k7RMvBELR)hq3?>d?9`Q_NI)ayu(iI_MPedJTCWqs#mSx_kS-aB^g55TqzuC+U1#}YHz2EFTMs~pSZdi-YS(_v1C~biXT3d5x+-&{2y;<g0UPEhgp(|aELZoUM?D`~rq?NdMO`P>Ol1IlMc^C`5!lqA#johur7;}G*BbfaOH?-Wo6T|zPO55d&*itB88@umB4a&`Jxu1OKbZe@cUoskba(-QZqR)MF78(*n0c3Wqzu~!AgAc;7ygS~X0*ip>(%4uq##-zjMfPuJ^S+K5GC6Z2#aP3ytG;q^C9L%HdpFlMyg68Lt6Rk)ltNq-*MmD2G)!oQf`JQnx>lbczW{4biO+bFFwcQ-h`bq7qIOPfdf>RbCYm8?$#AEBPrjG1)41oAU^C_ZuJc;n(%S&p-_AhV*6LSYk+Cul%;<R&;O!N91S#%`V5x2IRk@t5*q~A&uYzeuWYcpTWHd^NbFkStJJRcI+a!&o)$6wQ{83W{$szq`wgDlk+fVqYGlxobsS2(Fxa*JitYbR1Tz3w0@|E)C{%Cr0O--t+-e`CbtNboAWTPwG6@lu5GtAV%w%V24OF#||EVQd=u?WsNKD|=aq$MklfSgvUMXw&b*OzAa1r_!89!pjAYZLKoJ7T?RQW*=}uJL)?TUTDM80#r}Wlp9q)jp_s$B|Wa0i9;nwvMrf`Io|~4v2Y9#G5)BLHiPL?1f-99KgtKiPt@hC);WRfEGk^*VXgQIjJAniAY~538!EAYO}3sF+G3|l|bV_&+mxtb<{!+%=QdnQRzQyp7?+APKFxe8oQQ$4W{2pcdH3wy;>5x*v?nwG>u<p^eWV)<J-G_&qt@m>eIbk4JBgzaBT8@fG*f7xkU;?l-`#QbO`p=-F&Mock>=WBEnYxb`U%;1~ZA>a-;fyW1scxs~`3<Ktw>aA#>e}t?yi#9JpmxCz!9A1YLdDk~@pXbGvk`_1r0rY7M+k2qeBY3kvEJsIfowXO$^ceg_)Z9UB$t4a0+_)Ap>;-}D1pZdasbZMPM-b$+=D=~FiHr?{R=`e>hx1+hEIoAeXPcOq%2lSxH~4lRFL!*~AUMxKIS2Hl59<!~PW$oq4s@0Ng|p9C-3^_rD6-psMJbmq3gRLZ1x*<OcAzpa7WLRr%ttNH|d>aQfaR?R8lS00gOy@f^bnVG`i`dkx7o2LS`tGU;_gI*#jr+aTy%5m2wUq%0INN26RuLOmcPqt63`fXvOF$j_ImVZ*FdNy*L?IzuAZN$dO)$ne;775^s;l$Z=?>&e*^(tz6@v5sm<MA4#DFUajvEM4_)ttQMxYK4vN)O%3Nz!}JY}x3xeCyn=zRG5PeF?R^-Io}RIEMYiy;Drz1q68=o~7!`2nur77C5bisGt~-$NE0->#tIOvG@{OEp6P+CcTK=+ZI=Tm`~i}^++CCU|DMyi})LCQD>8&8F|)2k{#)_;&<>KqfG8>jze@@^Rx_m1W^?VyC;}?-Pq>q05mUqQ!L3PQ0=8vUTH9R_Z^Om{eWoQ*BG!b(UvNJm!wbX^ri9=PEHSx(o@h`+_vl2d)*qWG_DI?x41E7@=@i7Gtr_`%+}WBlkgqAZOfn&6CU+mF)BBD=*t`4B5r-unWW&L@pyQZhJIhqe!W+vznZ^)+Xh)s_u3z)a;x{Sb2AA4TTbBs=HeB99k}E~6}YPi*(eGS!RpRj$Q4qY_pj7xf>X@k;C1#GObs6B?G*YdFQ&@A7POBsngPD0lj<&;ek|+wj+gayS|Q>4Yuae**<uAHBdbI5b*=>jIsFd%9!fV<_9*viUgdbVo?Lo4)ywyvoJMFj-(|oXe(nFhWPW`#DOc9KMlzoN*ji4C>|xN6%YFzpaQ$Zd#p3hSn_h0Wd)tJ@e4TK9pUgOamRB|F*o~(>K*=j}W5(cCTzFPGpXzfY$d_s8^{H;M{#+&Rd@LflX)LP6DSi+dhuihTM%(1rtA_Vi`;kX?pT!WY8s~NYI~z-vi@i{V%<5+xO=kMn8!nIdea9l<Xw-f+nY*Qg96;<B4XPp-s$#|$PE;*XNkxrF^gNp3UQgRpuhFef)?a(LH6HahUf-pvF>1LRYQ*AYV}n#nZ^O>I@fD-JTi>_BMRaTrx*#-RWK3E?m7epsH^ry(Zcwv^Q|>O4Lz#1%fwbz%?frr8E{LxBWP!GHnJVO|J0xu5rgU4$U)$qpNIjP6q`gh9f44GYZtI0SSQ5ZtedN!mWmazk*M!#~Bxgfj%|zlSU%o@_V@zL=?PAE5U)OeXr=&duTtA^YgRHH2EyFKZoN?ho@g`i@(&+;Pt-0rFlF`{>9-=zph~F(sZ0d~Ar(WN>HUhdjHGs#?=>{GMHit*UDtPbq-RY^OAN)^m^|YS8yK$Lfi|!|8=ku^PDQi|G@DxRRy2$bAAC$vp+;Fg(T}c7sUJq4mmr!nn<c3I1GG1tB5Un<s8V}zLdHBp9?c-DzXPNd!9?dXXVXwjV*NyR~^=Ix9y)Plf{D}y|Oa?bB!rSgqwXP^R;lt(UlstWEo1fNoA}QI?W=djHyfM#%Y^@*I-)4KR8p@`!8c}aH4qmaD5b=H2@aQTtVvWI6dh`9$@Mu6S>0AElfL>K!=$GbK@NT}4dv^K=7&)JAciHHnAV}P1gvHn0_*}cEVvXi;n?tpwFjf{fvN%Xnsi*HB6aKH*@LnoRmfkK7YQ5B@F5qN{%(wBYxh1!?c|>Wldze_iORcrhF=6Q4*r9#AxF>LX#b2m1y*;*f^hACi@GKc)SAW%DH;Pr;9!}Hj&PkwmAa1>b=!(2WK}igCXIe{y%g&ObNGA!g3wCu>W`Om@<wG6<O}9Y~8Al-2dj)6DkK>!xE*?UR%bOgMyLl3c##H!A?0O37;GufDEQ8lVd<;h5U|1K}ZT&HRw~mB;Jhzi1Qj5?|b1Sha!04`KuOG|XwV$ASF4{JhMtfYNz-p!E$EQ}WSAoaKH$4gSI(!nkvojqdgIIzNX28jd*XMeOAZ~33Ggqx$r?qzP01QEnfH?#|-(UyK9_6;ow0m#+DEEJZQ<hJS+sM2P4R6<4P3-j@a0<e&=)*I7Z_umpU9S{B7dY*A7x+j+m8^~p)CY6@e~6AjWbJmfPpMrf&oCJ|xUEYQNpE$3AEvAMEd!?8HnRdC`7PtwwZU27Nj;#j#+?S|l|HvS-+=84?V?((KJg;J^Nn~dsox^{bD3J&@o$F4(7heLWl8jC-@l^MpQi3KSsk7fNd2xu>s`^ittEo_><a1irhq*irM<R2NNGmy7*1R3uM_RK37_r)Fp3s?z(+<U=|g-t{MgpDn%n5Ub!=4KM$J4*#P;fpKlsz+_#clvYbYwo8>Q$62-zGh&W-hsciJTe`hBlE+_WyNgKcHz+#F;)Q!%c>@n_4MyEiy00RLN)(Q0Li>?j*}_B;Cih{>TA38%AYX^Q~qi;9f9R-b#8gyH>aIxyMN&FN{a25VrxITu!Kkp1+<RlhQ+t)$1pX1puodstnRmUh+rHjkBWNMmJ220w#SRh$316wqxh+f?h_awToZ^<2C~gBP0i>!;E9^~bI7_&dAIA4S0doi#?jT>?}GRmD7N%Sif=bu${0c{;uohSla#B!Bcq{P>l8I93Vcn+zfi>Ltg?f})V%`*WZS+YR~CX?(>B!hYq+62FdH6HuQjU67r<=ac7MIbC1(^}uWZuRgmz9#Fk9jw;`p)IFXZb;|6<+?5q>!|H7{|8X#F|Jq2CMY(iBYa@S5oUrp3!55QDbAixpqDBdBh*INejEz;;9@x1y4JOAP^+H-CZZeE9<!jsczBxV~ipEnse&gr4lHvRQr1g5j3U%1n3M+M<$R#n)kpx@N>Jez;edO%2g&4hoS-tVoaMi{3hhy*tfgOCn`ZW;Wsof`Z_9aQtF(xkkbWp!gH4{Mplr6l&JF<nf`F`g|a8>~?DU(KBMB=nWlDe04ZDECGWnb(eNvp?>wX!~0z}1Lq3i{y?MmueKe;{bRwt#{n>BUXde2jIy$-f@6S&j_H%;fhDrvtmFUnjIfBmUi2+PlN)c9)oir0v)22(F9Ey@gH4zYSae4c~Tu?qCmrD1o}Wr}n>ED>#{--kLPYsw`s&Pv8Trv(AEX<E)Nv?*hXAlD#r%O5bJlX_t`Q^H1)ESSVE~<z|%!bO-hCNj2Qltmz`cbSX3bSa{gRu3wpYdoZ+G)5bIq$w_}uyi8RJ8yn?bbRW$n&<=oA2%nRC7H)p?Z*z<<YS`{;#>EF&F+Pj0y<07O(!!*1aAu9lS`1aIqg8ks8TNacUGJNkNp*_;iWDxOby<)Htog_MT11MO3+7f<y!%Pb-!`@!2)_G@M@Y}~O_v`GX!l^`Zz9ZE7ZG`0t<5AFHg7CeO&e83=T?~urw4Kn!fMW37WI?!;6u3I8f>Xg^P+sfpR<l}F#o`v{%HRjz^hJ|w3xpH!pa9CCAGe`Z~w8!p5+9!4QsF4jV0-Y_jylckZVEInk=U(A7cmb<TmaV?%ursQJj&rt5`qy(B{JcweDlT8tb&(T<Dh`*PXbf^<h`HUqLrk+twlr4p%%pTt7#IJU`6C9>7KLv48|C9eZ43ArYZ8A3+qcg#&zmNQ+ylE$Fe{oDa{pKzTQz0bZHk%xwp$O1&Dq+dr5Xzlkxts<*f3^w-;zV!iOD3wLu?HsM-+Xa$*$Y30GF0(-E6`V8F6(wbgW8=v%J;B`{LnPJkZ5}c`aV!&xJc+jFcdGlJx=exXsD$wYy)#_`h3Alr3f7+0I3mw|rd3-FM>pf+6u2gGI9PuL$_17Qm*VYG*za=rxpJuo9Mn84ik?qkg+L+dNI(G#xOXue9s)mV6!6&0ViWi7oAR_#(i#-pu`1bfbC(Nj+j4Xt_1tTMURbj>E)+416MyTezHq@qmA;KIvm`PN&vezFx9nkBl&&z}}FY>>FiHWSf3Bt@X1*6AA*>kCc&0ER500V<Kua<xO-@=pS=Y+rX(Dj<{XyEv<;L+@}nyQ?5p8Pm)`6yZoV(;I*pK!mt-Ozg?J?$?<;EQ>><0)h*v)-qBYkTHB<;jX+>x;_M%Nh{-{EE^QV#kc>#81yFN94Ux^HTE`&zxbKcKbVQ(=CXkF}Y7_LzY=A+y$2v(eJOU$yhB>t@>WNog|UQcGAGD=XVpm%P)6329Lgik1rp4QWGTg1PuhhxHSr)9U#B9J3V_Hm=N!a$*s{j>yz8Y!EYWo+dBhHHN^sQ<BPC&8w=ybt2En$mwbT`TxT(NF4g%9NVO=6>dPhF&3(zQU&U!eF#+K1bZIn}k>8Tay!QabMB+NY8tk0A!$)M^jk7vXVRlUksP_(g(}mi*>4vnvgMuIaij%QC=s^>`gR9z{u|$H}B2`5*w!=%;qX%<ktsUwD+(|*9>lb0yd0xL3i`S$Z%!-?;a5w`8<42*ZbF)3Iz>nVWPt*wPP0f@%?sI@TP=b1U8ho<?>;Dn55&+tI_ikPftFTvp-tomE8_?@oq5t-9GG1`z^ai~!O8rrTQj&imsA=qax1QknQB{)*VkdY#;hufC)3Eyz{(*08%#CKP8LJQ6HrE^KacAtCuiBY*UpMf+91PX^=+F=5F0Z%9*7I;0Z2Begm*vyb-l;s0&vG;VDZ4-J)N0)1v(f1{umY-z)05zS&=Q^NaoJh+?ayvINHQbZiHlL_B|p`r9y|T=z+dM%-0h))ySu(Ngi}%%;M!nVsjOGQ4@tTAiT5@3iUm}~3h}wNW^xw4kHauCSLJ1NC>(AEH2BFPQ<od-{8=HJ5ToE&@@8vewcm+I4d!;3{-XW$>i8Vg!;i3-x~KLOGp`-J+)0|mKyljB<7e@3+*2>&nbX@^qkZcYM{WBfjn%*WGZ4dhvOT6W-Pt;$)q6KS_5*W)B=|#=MzU1T=LF)IC|~`h!y~A>zwnAS!Y22%CJ)4B+)zbu#~aVa2Gy03NA*0;y4x>dP(-y=iXC(B0MlG>_`$k@RvRJHVj6$)8x8&H4LO4W>KwCMF@JZnQG0#Er9nJv7exxKsIc#sXVgemMhED2=tv@$>ZW_#UQ_0NZouEkhLwMhsT?<pI@avAjx&E;c|wXhkgVh^h0S7haNQelmnR>9uOmCOvmHPC$-HRqg=xCI7ZhHNf!l03i8YaHAVeI>nHNo(^Im%5ELG0Qs5h|LD{;K;(=HU;`G^92sW{KTt8HiS*Se$GC2N#dV&=9!v)Yxkpv`WI+G82p11F_vm**RM-WVctMw+3dCcb>Ek-w}qPY&5R$`{Tbxc(;g=svLC``wxStEt0!+g2*Kku8nsYywv0DiWN)-t2+IGBrCskU>3N9zNDmAzu|B1WDAhe_lZf*2<b!V0l$0*r9)Ju#ccc>&D|#@$7GSuf7^vXA+*K^j5I<z2<ly7}2v+cgEXI=Q6XaEisQ-Q0jHe^!!#DzLxfH_Bf5(QRAzve6`<jDgIm8jI!ZN0dae{5<OKhhNot$ek0A=qnL99w5?8f-WvCB#i9>=2EuGH&_5SzQAjfd8(wa2-m|WZUY`XWb;rzJr1q)7F5P^zPNX@j7@K#~yP-{ed)jvzA)fl(&558o!_^M{GnK})HCgUN>rlsQ_1dShXS3QO?!ffvA(MRHx_lYjw7lTjZuUk!rdd<)CW2mm$AUq0CsU~Q?yoO)!jQiI9*wsy&%3Sl{}XU9p&F_Anzth!>0T+i*vLf(OkgO&@6%dmGX_L`_kDgVYL^?`ular}G&hW{;-}D)v|t<-Ygg_{xXX~vY5KU2yc=Z2IDg@ru3|bzy|Q!X8zOt1i#Mj`F)(}6e?GVs-+mL?Tysl^y(q8Y2YWAl#DUQfO?)W0v+6nWDR;302N76AuM)zwv07v6@{{ZN(=jj(2hnQ2ZzfN`OE#SF#Cufz6vs(To;nWKN`TBfh-GrFZ+iNJi%myfzI&DHh&GdN7buOK>|$b}&Oo9<ZWrT*eIJ(?+A!OX6GUd?*Uej9K1`X6_6zTIdL^?PSfbn#bYk@QnoylT9PU1Lu5(farolRCGg|L*^^*IX@_ES_iGy^#p?g}@o$GUK&|3jV{;&_M2@%4qpU=9}<(_Tl3u(?HN}0C@lj-s*(#*Gfk)5&}ehkzxYEinsUKW#Uo_o`S-0nqlThH3=C1^BG)1R%Pytl7FeF^Qx2QO%@WraR|<TMOpr>uN8H>7#6h2FXH4L;<)4E1;V8Guz$sjH1kM-_dE+U3#F-bP&Gm~*6qrgc|?;UD=`2l4N!J=E2=&y)8&95PlvaQThM<xROK3@6Cvi=G_1Yv;Pls4;dOHreFmb0=~8tZgR1)J~9z#~r)R)aW4=HTWLPdw<esFBWyjj*9%+JOs&J?v)Wbm+lGt+^!E)dSS77yzbSWu(zVeZ%JOW%Nsa(Q0&02(CqvP*297TUqoX+52{Zk_CXok5U&Umz@VdRRBeH6J9=;rz-w;+@(J*_i;Ctk-V%NBIXe3w*ydfiMr(8x#Y@kcbOpo9UL6^jPLUec?Yv2X@O>-5IeV6%btOo?+r?u^KA(*Z*g8B$*ZDc4aBUz}&D*DD$n_mS(k5&K?GlZeQy#cJKWfkpF=-o1i)uxjd)d2*$X)ZY$7Z6&92a>M^eqSwD75HC!_7&R9_OctvbEm0zF&XjYoBhNRf(VkfAs}sG{2W3De>nW-83{+@#X<fvTd7rIxBs!%ztV*-;0|6`4o`3IcIup-rdLwP$w=~yzm#VHOs3QY-@P5%oZFx!~JE}omCpE<MlTPu!Yxl>ic2pR<3Ho;?d`FwJPJw5z7(l28@jXQ>`p4<^Cz6mEL_)akLNYlsM(>=biW30N0Yk41P+boh?kvuohV_EBGaKfnl|m`sUBh^3~GLtoH&3GvZQz9?Gn@Sp}ycT7?;eH>bVL<JYCR38$|^wuK-+Wvdtge3}{O)|~(UbOgY8$<n(n+HJdLuW=jxa;*wQssr~fn7zrsukfv&<E#N?mn?<r0h^6^(VjcBIv<~wZ7|49>bA!SddUA$nf0CkUTzj>xLGOCpEmSYijK-+<CPC<{uSy?lZKaTOg~oHhQv18#FKm}TQT)R?6Lbc1{#4zf?MlMtH381uB^Y$=f)1Wi^^cLD5LgPZcM@H9nV&m_cq_q!#OX0^FwLz?DC0jvN5>qaly)1NLWgraC|I%-<?FSyiDEa!Dypx`hx=;zG~O#)B43TqdzN7TJPT6vny}*IE*rH)^adD`~_~MjN?4eFG?_OjjBbO*XPD%;mg&dxGcyhfZE-Q>dRKb371LrcxT5C;{o}?pm7}_lW(6LN^;V<uNUg*_&JoyiV0j8Ay-oPNb2PCW~`(AGkIx7T8x<8pAc|`krF<0+>KqIQQC>%qH61=|Is(Am(tEbsvZG@S$g{##5v5Q2mVGfHpLUom0jb4A|0q(dXjhRQi^ivb=(jc5L;nmg#dqXu6!nmFJ=$FSuKhIm{fM+Ux)4m^EJnhW?v7*ztBZi+B0|BOr^$+f>KnU?61(!9}thJ6}m_HPfMblW1ZRUN0+(sEs1@mf2XcXobU7d7W0ocF?R?8FX6?g*<?2}{_>)&3QB7Ts}2_sf0FoMe(JWTfL>`R3uA&dwOLSIr#IkyH7J}s&sCOJh=x14Tst;vf1Vhp@!9ED90!TL=FWZZnk=_MKc&6Os73C#HhfT{r$|z7=zchlK->E5@_@^dcFzDRMUE8(5eGt_pN+U$8i8N1QUDZT*bR9ajvmUc=sjx=u$c7wO~Gr$5hI<HRikbkxYt{xP-hnJt5<nJc1E9kjx-n2cD<lmS{gK3O={1Me6_pkv@EUH<BTyee!1#5XIGsiN_OnB|JkWpt)=YldqIkx)eqB6xIaOwS2&Fwecyu|Zf!1}>Faa`XA73RH($9E$sIzkbWe>L4J@=@6qWaCZOmGrX!4i5eX7#sF6J_ts99F88lL%9bu9%Q&YC^wxaC6P)O{=_C$7j|5;=YWwY(VIfqMn3)(gb<@>7Gp*KXV`ewN#fH9daU%BHtGgT<aYtzUygd$ph~0#oN0cTdfCV+{%i=TLiF*9uJfZsSaoBXS<qS+BM8$zuqQ=!b26pZjie@no3l^h!&O>*fxqWB2ub%xR@@X||9k^5aADBo@T>vY$Ps9D99MpP7Jb8)*Ssg@sYS&9XL}o9^r0qtgC@DwYl)>1y7rJdzps+A}Ckw`b>X)-@Q280FWjZ~bLnakz9tK^|5&JjzVp0~6;mS8<zyU>dW;`KvM+(a*@be}0fJ2c<c0&`|wbkNsq`6R}=y>MqewJD?s$t5xSF#z*G9c?p+dq)8_R^1HpYcz>oX2LiVG=*+LM&D`&!2_AQnY`x(c>s#|YL-gumK3U#})BC%q?A00ONAjVB$4#MuEh-9F8OcW%eF;qavSaqrA{NeaJ~*G%;hNts67S4@P-6qPp6IWGKT<m4ji+s~KbPK_(j8`n=nA?A?72wLEm(--v84W4fhKc<uge_za`sKS_?`Fv%&Bnergm7~vwI`lo}osc81viJ_#W@gh14eLcfP~B^Dfapp>ed1vUYX6?rt#k9PZM_NH_*+igYqhb}Qg+t0A|guNu%~$OmE))mY`*Gm;UK)j)&elDF+cc&M+Dd=ZYz?nuo+9;m@e9qXE!2_B$jL#XVD!+X7zNLsqFKOgZCabLve2_+$OHGXX9jspS+K0J;|ylaTBB<fJDr*SSuexRUba};WP=;cmW;TySfOVy{{CM}O1&D`H-UrK}sEsQ#9e)t5dOizYaPj65Yq*=~Xymg0-+d~ZSsYN!68eP~j7X%;w+3k6Hv735LZr#T*4c#*P`lOZh;%>D{r_q*GWEji`B>`F^o}CC;`z?t-w8p0d_wM3L6_MttjA;+e^kYy5kKyC7dZf)sue#5P&N}SzOD8^d_wVQ8HLGb?0eA3AatN&5<yOhgqkSH|YQ;Cu`K4Y-1UOv==gY5(^*?RA-Q6HVm(qVvmAO&RHzbVv(qnximoo_Kw%B8byMZE*-#oEs2FtVx{PuCBbA@NVvqr9gW>&V(@i!tdV7N^9m-}enS4=~gL48AfFxLyoo0oc1SD?=oW}FKf;Jt$GKjrB$0?|X%eqF}#`q{k^y<xFq#w0jg=^0a7Jf)ikHhKe*f9*F%eQ@yar8nFFi;+=RE(7^&6oi2e12dt5<YRx|)6?_u_~SBQR&JFzrI{3tiJ3h=^392OoA3Mn;8&i`A1&M*YU3wUJ<d-^zs|{=U`4Io0!i+xq93hm5}O4V^*Px$*3WuA9xX?oFnNn1z1zf(l4^gn)1cSA+MP~ozf3md-v#<3z4-{b;NkPBPNF#NuWkpRG29Gs62+4Ce(4+3rO}w4TKB6gANs<W>^z#til?ybHg9gcOYA}KdGFiY*34gY68gKd^W`pz+S=9YnZ9pnGtT2wklFXg73UTsY3~MaO!8B$p&9nl)}rU;51{!-ESefJygPe4^?CW&hXx?+TbxDdx5~BenrAP4WuXG!=Be6cwj4NlF477rySvt1j<}k`V@d~Olygu!rLHAFz1#Udd+nP0fULdcD6JP6athT!zJ}#mBz5|m%c&SeQat+*=<a?fRvGVK-Y&|nF2%89nBxsN94CC-K80P1GbjF%PES3jkM|b)sl2%?*S%3Kh<gDF3Dfz$ciA^8sMFIw-JHanR%!wLIH92YWuqr6P_#>H*}1;IgjvPiT8``uY<XSf@CfLTmrd;$t{_upwlEwTr2_PUkI<O6i!N=(9w5$73$hOPR0><(w#a>#lo_RdHdp4|yE=mRYR)b4a)zAE2;k_|S1QlwV_WScO4d8-&T75rz7!q7gbueFHW6;vy6+y63xmS+qz-IA{fu<j{7Cm(c!Hm_V_&-cK^t0B8^fdpZ<l=MiHxZKf5BQB(Dz$Sx2`0J*^!XTT5Sdg6?ENKpkp}pW<U*%UTRBT!tEZlU<jwO&puJO+-ZsOuID#3$s4IqJB<bAv$VszR;g<-i}eQ=pE{B2Smn2i!E5EmO)dXM#hMC(9V>i>$%ljdRo$2|L`I+aLDd20B0Z+Q(<vM?SwC+R@5}13SGPhrWLeW;Jrk>m6iwIXb%X0%U)rpV(ZmquO<z1BJ?|nY3cBvDO@ugJds-whh}Nd1A;S%m6H>YM7JYS=L+tr<3$6R&Siycf=_7~8=rqE{xKe+1e>3y5fR9SN<?qW%H)_$-m)96q_SXm-Di?3~E$K7xy_@pPtM%QRY^^2FZameaay1Mu?sqG9I0gIK6{kAck!R=o=FAx>zbZS2!^hx}y0zQ>cD4HcWe>u=>Sr>oPbuls_+y3d4D9+b<+ooak$jAn!L;!!Gj0ry$GpJbKhzns_m2q@Z(1YsZFT95hVVw!oz>CvaICpJ<}CXtc0IO%4?mr2DM}R7r5gYIGIX1Ju~^)GB8%|Z4O%Jdk9jew5c4|ETLT~Mp8Uyx0pT3JL;?-Fi`Md_HF+vHDL-ckkM&{P5r_APv$u%8<&O;%BuBZD3}$xk#mWYc8sQ(W{EE$)8{{+pc367Dp1zyKCubmjN`BisG<$j`-f82y48_es1>Za6?@qybG>xPCbm2asT2x8e)j>d0%k~xOG{XVSi^bae|D^7DLcbABq?)8RJ$<x;MWWiI;qOv=%Np~+dzEgw!gtuo$)9pU&&GVAZ|2nlIWOZ^SYrqjJ6Jn&BJ5>IXA9kAk&iEK{YS0$G4Dd%YUo%cq+QR_Kplez++|k%omaN#PoY3OObW57shwfSQK#qxt*`Ofd0KYXg2^1Av+?z1O9PFQ?Wn)j^0K;q-m8at%0G(NUn=F|3xZAz+8~~hZ;6Y>A8X=1$&e1zmaiCt#?uYrffGl5!o)D!wlW?~>rW`?J%h&aK#k_RRhyZPYIg8arcb5dqRi-)){5K~A-wx+&BghF%GPWw*7&34DM<1CEx~NmAcgwmfZDAUW@>SkY2c!raXEBj$+b7OU22fo6T5-++1#E3(N3fKwDa;W&)rF7OvxQXEomp8M~&t%36vYVDu=YdBNvhg6~j;iwq)TcF9p4Hj`P4ZUk_(h43sC<^^p1BfxQCM#{%gpV_9w;R}f02o%*(0RSD7Wii#D_W?o#OJSpf-TdmvVi;#J(mB^K3d+Tz9Fu2Zo&Es&mo>fY<yrZhk`=r3%PU&sduI{)4+8hCf)H%<I&v6ueKCi~;<qEB!JC5D=elI&`n00#X;kr8RHr(lT?PH&bH6uSco!=i;QuKGd%`89-AX}x|eh#iSyRvNWOF@N~as49}+l-rM`8J!0=cD@^M%88e9X`&BHOqrZKD6R>?9kA)*?7Fx6QPbKM8h(_!+>Z^FAZSa{%hC{@L7FtHwe`k5&nOMFc%bU-<0h;1n3RCh2EF9?Dit6Y=fMr#l)<raxvUYrofuQQshdUtc{EN#Gqujx!mqgGF_k5d-4umq-1a~(7P}K%pVSa`UzNl{yy#7Ean$Ou&Z#7*)o4@j!V^sO02*B=zVDR-tN?DO{@6T`s|A^@+z;lHH@awk&~3={#L8?59h+4VBxP~85iM1TU7lqP+s%OaQ3UI`vIFRTH7XBp(6!w&GaBb=k2Nn)aGzHLoy-QP=(?s_?Y{li=;a^2>B72rL0@Em~GH0*H=UD%Qo`)I=u#p>$G&qK%kF4Uml`D-IbV~&QWPd4RkZuGBX(-!DD0T6te~MXLg>IVqP*uH!yr>CR}ret<b0+FPZs$wj3hykFF}+tZsBCpIbT;*H4X?7O%6im-ZA^2cJbGrx&``VGmaGc5jGljWmt+lLj~E7#TQ)N?1dhyo`2iEoX1u=(~8$jiX{Y8-984+aB2eal~LGSDwQK?-c-a;P05DJ$x$3K980*C3Fr`!f^R2@g>%|0eH`Ypa+RO`NTVYz-Pq=ifCAt2e<psfJ(ZAN_uY!TYqv=>#Md$$VP9?BWMVg2PJ2@v=)8C<$f|QWI7Z^^GWj@Q4mOUUM9~$H>i&be&yLJUH=U3Y-M`8C>d>Rzq5ewnxAG%9FyBTHLFDxNwm<MNwiXZn|yl@-V}2BZ9+|H9A@%&j|!(@MfFD2{p}%m0rDt|_?{ZK(RlY(WUV?8dXl7PSGjjYBj<M{v+IzGnY?G*=Gd><&pYo(2wJ0Xa1hN(b$hQ_D{+T4^(UtuXJx0(QE6Ql>GsLYjNg}8yR2$8{=YTxk21P$z4jGfuVaR#Z3B1rWv$X3PU%~uoi({(bx(%7Y|GJc-2ud@evbTS2m0~U*zQu#qPuJ_n^x1=)=QQXxK6d<FZ@RbgsblZ=ycab<=NBfg~s0&KA)7D`ZYhNQ+<hZ!}*+e+2s7MC>|vS)Rwmm$H-7Di9sOkgA*|5ULGOJD0IHrcaO`z1Q#ZYxe$e?4f|E_552ZKlW0ZKhsDCOBi)BZF%unw0xPA<k%kuC9^PdFO>sb>D-S<9$Aljnxb_;fWL?>s_@$bcN?8j=33i~G;roGQdpF+RDU*`_V3Ie><<$Rj!RQDq7&7M<Suov|?GE)tM?{L#4Uex6|3<~Gq^1XM?^~^FXIZH3BVoLzbtlw7eX^r8y+Y$=Z9RM6P>mLpoFd2*UI6cupQoVD+%>PMWhgd#OViRiVeik(8;?4@nms4_pXpsr&KG&T-0&cBW_8I7^?cT2S|Pm^TpI?K>2#;s7*D9v$6Q*s{w7vXCDO{y@J^L;RL7BIvRPTDU^mzc3fmHPO7oBVejROfT6Cw-k=F}nKH>g0#XK9+&P?09iTlAN3#W;$24hpN!bT~k>~4uJayjcy*vHpttNS;j$NW^i4lY<O{|T+=&?E5>p57He&DQM$#>MYcc!Mw{U3b%Fze_E%O;EWan{nvL;HQ%m$h7zRJ#6KjsO^@;ZVvM>UE6~%MLyg9KE<cUgyJy(K~^6xy_wjvGhCp=3Tx?9PhEU)x5TO-GhtL51jAB%7Am)uoby1nVYS!-O!b8NthAw_Lqt;NUWGaB1gTv+%rGyyeZpZIyz{VW69Z4U<cmgY+n?$kxhrm4q3O4<G_S1jR_#5c#p>~qw}?L1j`^M{tnR-2d7oS4tO9bcOI^6mchy&QGgu1Z*CNN}E}V^Mh|+!UBk(}A>-`bZ`F=&qtsZUq-}xwXr1tXnms#xCY_`MA^4I=T-P3Zr!nCtpRA)@D)+w3dIlD*CoPe0-ru#$BBeeK_vgCA=;5?c(E?^wyZS%W74jZX_Bx$_r3mKJdj>03g9=jyX%Ph(VQ45yF^<^dda}C&J(%AV*AdZoK|Ao|BH_<+tK(a@Yr~9Gc8U8H?u+n^9+=9^YE+zFmQJjp_J5xs<O`|$H1UGn*cTCgmvgP~z3lYiHb>ZyNo=H|kzgJIPll0%+!A>;V^Gy%tW{&e-FNe?g0mq$cqDgJ95MEdX?<C#Lei~SZosJIH%&Z&vV<7At>iH@_cKgoVz4^<pVQc-J6W?8aK1v4usQs%q-%*tAnM1H%)KxnJKGJMY47RK_*FF^fFvHILZfKKwe>hh+yMd3$w;wW;$IWa29BcDLIM;#oXlGyZ`TTPc=s2S8J59I3@|B;Gtgq~byY5@G%E8nX9wOSVj$O0o7wy{ph>E{s*Qkn}S!u(qlJ1RVW_4XI8JhMx6Qot^jjMT>(eF)Xsm-<52&$A;YmA+K(bL?E+Iu@!J7fc~leQ1-19f||2lrm%M(hJ1c>_4R%-y654h8+voDN<$?b_hG@s@iN6`3o<_bxs4o#>rFGg#_%MgtF~>$Pd@_TQKIa{6Sic51`%m=a^sIRNcP(tbn9c@nor%NZk0^7QoGmcvgSKD^1tl=^w~_P#ki$I)7*j05Bxug%I%rfk5-I)m|7#6M_ihvqc>6lre$kWR432>STGbcD}TTJO(vIw`j&%3*9fVCMUXXN@{|c=wsOn9l)~tEOUa@D3U9(5}iW$q2GP*<LmT%W94XFx`0^vB8DTsr-*H5DsYZLsyv!d+$>&Y`MQSV^IR*&B~jv8`Sgi6i16Sw1LJpq!8*%iPy3|l1%l<e2d4_8Jm4_?{xEZR~)zTxZ1Y)v-!7H3pPj7+w3C-``<(R6SDJKdVxL99Ddu<@Vt{zDT;UBNmwf?<^D5ypGgJ~>#e~dd_wSK5`FFQt+K?M{mv}-%<roxVWUN;#{7hZh2!|f!{qf$ymd#M1I`}U4V}xVXYB2!3b8y;<B>Oh5aTK6@8x>ua!DHJw`%K7nHv0vxs$732}`Ab?|aevEzLT!?Gwb@N7)0YLsQu2wso&2ekfkz%E<j4JEHL6^E&$%qQ~1JCp1$}-eUC}4pDd=0e(#%ow=6>4*K}dW^LFyiv<}9B|W)+n^m*x8mT$$b{^pOamE+n|C3Bhvj$B2vr6p{-JXpTA+!B-ESxg%(3SB{Z_sEe_0_H(9o?CF_tZ-hdabj*dHsUyS#9-XLFd%(tPtAqL|1mL5@5t^&vTQ;^B9~a(|q&}GSYO(kM?UFd8zrv7x40Nt~VY>T%p5JCCbJhF_G_cKfLLNUm1?)mq8HW7vl=mNKM%|H{`C*16&;+kX7n+!^6s1kD%op08TTMnf5Akf6iQ;Lq+@~v7`5ivCKlB<GO`Vcdc(PqSp-N!whgzV+X7ao^JYn`7A2#n7Blvt(lPfIp=(wsj7cGFMfEyrPd@{kaYCoT0FQwyCIQ${@U+Q9h0li?z}9bIapift-U%Z(^*3RtHTBh;|qL+*#mj*t$R=LAq04NWMHZf+|SNcy1|`^QXzU{uD9He*mHcV(ucPCY_@h2bG{iG<puI<mDZ&nF6NEF@C<#+r^IGZ)hWKHJeY#Hz%v(RbhDL_2yEHoZ1uVZf_EvFaC_G2@+<;FOslf-N|*80cR^Fh^YD3Z1$N`p1Bb+w&|KqoXE+b^rcX?`qRa@uqR46wcGxC3c4+~fVnmaEy*|e)0F|LrLd+ZsKjU|8({P2!yS{Fb-63~O)``-6SZW%ix-;5rk#nW0`1-fLv1S_n7B|9B=%jLM^NQbSfR#I%ac7akzQ2ffw38m)K*!3p+I}@ATVVahfNJ&SXOnu?T_h|#Jns0#aWzbi@JHqAOR|Y}SDfXWlhGQUunLCHtjQ&cMNR3LBd%qYA9&eEldtbTH}-BCKfqznZBJIMIPO64dKI-K`w~v?yQ|z<`89f5{quIOv3YUY0;`IxtP8XnqxZfhwK-1MLQt3Ym}u2W$x5dN$JMXmQ+aLlJh0HdNX+Zg+1!gLjBCN~q?hdhuXNFGP~5R<=dAk!!@t@r^QMSg@k|^7Ncc3BsCXbAq0v4Gn#sxvQYw2*TNC4*0H&iOjy6)&)iB-n=65_m35@vr!R8OstaD7r=SSc&Z<N2Xg<O}|>h(9<PTf{-$V!E_H#%)fMz`a-+<d*3yAR4QFYSvbjAUn)zlW2;ZJ@OU_F1f(<a>zi@r~IZ<J9Ul*{I~@c>N<K@yd5^(R*NsR&5JjFm_cvUoK|SODk)VJrlUUdQW%<`qmG|kiOhTy{c@Q*65UyKA<IoP<rM#PGQjOWrRra1k5EUoma?+n*A|(-<>v+JQrKF#-{oPgeyx>=1Lb>sW070l0Ls1wR{ymnm;Y>KcIk?o&$gVb~~T?NSJ<I`bk>qqvm^QwI?spI+0$dJ3ZXs;pk?_(9s-RY)6aZ=H{JVJH6gCELJRqZuP-RockUOKK!NC-1Pl#ylz#2*-^?0-}77zoqna-s@B%BpZ}VDM~@B%FreFLqX6;jM4o@FH`8Rd8xW5H9S~Pu;+^+)x^?iwY91d{>b17&cN}4qwN)yS5!;7&O4uC0iJd)IaFvS(wPTp)`l3I-=Z8kK6GN)vMs(A*-KJF|CjPNU0fQ4twXmciv?mFAe_wm^Xc*o%%2;dK?|&SfY14Xa9ELxOX_?fRL8u1Nri4-<dSOW-TF_>xKKpb3pLb0!nz~%S^E?hizC*zid;z{TKWOekn{#Jgm<SBD?!)o^V4h^=l5mgZ6z;D2a0eR2_m_8x_#9umCq3@WXn4T<!m<NspxJ`N$`@)}HUQ~{q?dWz$#0XOiu!lyTGf2Nk1w0ceXSE)mCdr&)lTGXy1D^ylQy`3|Ao2qV~lHKy|Ny!JJ)2!Lt}niPs%yc*&TQ9AQE(-R{w}=jvoImOe_!m>;C5<t^T2GMJ|THf6nsKcKeFUincU@kMrriOfJQNFvMYB-%FhM8FwXvt5VRecduRdkIPo7Hx;9s-`Uy&K{_Vd+tlRV>x9W01#|xT`)Dt`KXdD_uRva28@=PDCZl1niD`R@KS$aFfh05;44_nz28fS^(CEeoqk884wn1XSa=d2~$z|{Q_fNM?CfyyO%kt<xB|KkNck`s>WsJ>U&M;QVp0oNxO3wade;^^HIsB{4_Ij`rL14;QZNw*tu-2=Q`|d7pamh9|ro9?{+t-MyB95<rruP_@Jpoge=@PjO9^@Lyzf}oS+~1qKKC2TFszv8y+E|HyHVug6_C&RHF)?XBS?@ZxNV+m7S(?_n<RsrliVxmlKv0vZq@jGr|1o<7z3^1^b=gPQWKzy>u9(8)-5vbNxC25*sQb6TwlW1ra%(=IqwBU({ImmI9iYtpHJ{3$(~+f)975k{wC=QQ=w(aa8+Ij@FhN^NkND+x&KLfur><Q-n<Y`FajHTei9|AXUIUvfY<HF_w;Zg0KGLk%EB)vtjUWWvS$()_m*%-Wm-UkOxPvr#dadGT7ujK~TYf*imKq>B`A3S(Q5|kRW1IB(KQ_CW@%B7mNqe1C(R^NcLu>&`VF87?ZZPhM_uiWuT2!?LP{URU`QoN<{vP_r(_Pemo!^|*kE0$f+N*=v@4YUbBRf*nz`&yJ73A>SdOYVNo;8Y7M$U&y{or-UxSph2+hwQ7)32o5LJ+hS&A#$xll`N}InmOv8jH^YiIsa!U|oiIRBrY!?(#F2nmc8YPyD^6(jhV?4CSQAk|YJRLGab_bRsi#9w6!aSmM^SjSF(MoJ8~7-9g$>Yc`YhaQ=)>t~O3vHZ|%bWW7(!BD%5Pu3ds#qyNs=(!&b;*Uax+xgI%UH8VG7DZU8nw%lz^A3PD%x|!d5;gV8-O^3C!1?;z!)m0Ste>?0+B^T51V_V-pu!c{_?Q4_F0$8seEbcj}YKy;I>2C!7(IuOsm*DTte6d^Kt73Q|X7nU$;L@x5oNjl{%gFlVvVg@N(c(f0px|~Vrfg9=9w~q|#fK&V?gqF~i(^-+bgb~**Yrg_-+8LAQM6v_2=3N@o~l69{~0-bJz3LLvQwQ-C#u37NgOWN1G{qA)yX;5PJM2-7@Q6-L~|$hw!b#;h8}FPW%%jcVzOMVl$p>cr=8`Gog9yfkl5|`*Ws6&SLd!Y+DwZWay$E0M{qnOrwm5)7mc>FCm8PO+^OLA_$g|Y0KDdr78Y2ci|6S_+iA@nfQbC^W?6`?<-&#AbC*`Wq3Cam54_<6y4A_Wo~k}Ju&w`oBhrt>TT?7W^-;y#bssmqziR(|v@x)$J^6Sv&HC!3_szzN&<?#uvUHTbLkCT_lB<+n<dg@Nq#E<~B}<{n=X-{_IQI8m?mxx$wr$&fF+Ql*)%oy|=T#FS8<U<p=LsnK@)KXuY7D_a=k<EKuGhNx<g!D2scQr}<B?Iae<qKNeNCTD>}0l1<m1pYl#;MUvtMPdTCzd*^Kn(LvqBmCCim0Mdc{Fs^4K#PgM0`Kh5cc9=c5U_BA~rtw01vywE67;EZ|vL>MpSK-C1rr$42`z5~$lI>1y|8WBeRgHX=k<7uarkwHH}g3e|>NebDR*+g0aT4E;?$;o!-zt_4W@X}pW?X{SL$)U|Qjdj8fY2PdoJyGw1!XUH7Xayw&wLXr2YGl-b>_xY{mM6(sRkAwT_5ZKM0Jmvv(+UsGncH?|moE{VYxZ?u?&AP4r;4pJ^NAC}KO!9=~&|%+MFRb?+0Ew)8b&&FB_&3zf6vak2UpBlcQC13Ux#FutCMioq+#MT#ldzea&$cqE{h|Q{dEdCC-{YBA(kr-iMYG5zuT8dStC~a-^j2KzP}H{JwmtYWXn3wK*72u|iWs|nhvq!_VsCx=soeT7_o-ODQ+|1@5b+SX6KSP4oX&fH?^*r(LL!iME$VcU=2MzZKrcBnUGw*um+9=Pp)Vt6AJu9STbH(-njMSv2kkhc<^?@CxaY}EE^D(H%g~~Qk-lqlWkO-yIeu8sgF2sE%!X=7GyRT!d|k~K%j<F*<;=ToTZ@~y@P<4=U?4~htTQockD9p_0#FyagyYMY4+j9&Qz4XBURw96Thw6I8GaRG-FM`h!0<vvynY6)Sss+b-va%Pea(EEH}zb@Fu_N%K9+t~zb2ZF=G0eS_L|KAWBA=*d+TMPxy&zv2~W6#H`LuO?5G*sgD#ynXZKXw{r>8{)!}LH7LLa76JxJK%*OP|^Ym>xov$56!t^>4B%-<jg;YZ0vpC=EiD-NLmW8nn&Q+^+Gk@A2RXg+tyY|Cr<uNbg_WMR&y+@{cy8%@451Xxd`|<k$*o|I(43z0iE<G^YC`nUii|Jv`8p}72%Si9s{@XnI#Mm9X{ocNQGO1R#;V}#02$JM$tDHI)*V#`k$awo1#GtGB5R{(o8_I<RHT%|R8+(olKeb<5Y}bV4!6ewtAbC~~+N;$AF%-SVZZqtiz6+La{k&|Jwy)1KwOVf~n9)Uf9bbOwsP|{a6Y#3cRvfxn1)J$EaHi_gZN;rn0A7EFa!@Zn+_^sgs&(60G+kjZX!DEGT|6JhpFZOPz(p5S>TU{=D?YaFER>gm{PES4($3RkuJs$^U~Q7?3Q7oS@V$+i)tTLJD8}p1D-B!R{oPxXD^+kZ40i9n7nyEyzLMVL`hE&ay;J%2E2&S_XfV1_z#T1RHN(EkcsXw>N&R&E2sug~U%N-Fx4xZAKLIw=Q_Y!uue0(jQnwp<OVRUoh)w&@9c?COB&~N)vo(h*$b%ZA@=zroovV4jE;!g)+vEAd_a*}%^iyQ>7{3198QRVI(aPs-9)h8NHWU1?ONeY+tA*cfQH@#nq&yl-h$9D^RQdQ0=iFz^00`WgtD$Um*~N+E%=Mq4f#12O95S%6x9}yWs#8Z|!&q>@Cj>~P*62Np>XK4sZG(QPk5rzo7tLPf6aF!*J_sx=B6{z<p+3EXPFK)lQjP|&Ymnx<zM<+-Zs)FiimnPPStiX)=9S&_0fJcV^vdn8@M$bvQYz9vv)`3hF)q?PaiGkR&>Kx!jWK?nExHx{HSd`%3Nd>RaUh7zH8pGsleoMsEB&fmwSrnqHqL;YbZV{sJlxHtxdzox4AI^M4*($cJ1RTI#Kr?pF*c`b5eguC;JjGatN{Y-8SjZSso%X|a&DdGt2)=2T-aWEJaX@}0mkb6fKh{BONIxu(TYBJO|NFzeE7cJQGdZ-6`i6NVfZ(h%R$6G#CYy(yKtcOdnaJJ%J=s+o5@M&SNwZ!P~s)-NSdh@dja8UMePs{Mar?RJdM#2`u>_wy_)SBp8iM;=(m*UH-nu!EKt;0Z~O{w>eH-mFRfGS@E2$1xMU0`FF%#T`DsYV72>k$O2wjGuP0`QmTVj$&l^{As$bg)9g{divKlMF6Xn|X`VGi1!t2dJjN%r5lY>WpOc~Fonf5X!*TU~q55(#MSmAy#44%OlV=85z@AXF9vuPu<b?37<$fr)7f({34+8NDr&&`MM{8%3%>lUzGmwhDj0e!2mY4;1xuUZ2!HMIcH<t}gA!CgNMkL_FrhP(MtHpVXxUX=f?u3R<f)ADYMi@;0yj<L`3Z_Zxc{nOrP{75Q=hRFwM*aNF9^pmC5e#xrc{DKNVJ{F(lTs2x+UuhqNZVEj`X}lwjak}T8SoOBvqMfH}g4O%x6$D$|n2WFLcW~F~47$7lrn$Kkmzh((I`(b;Be$8&SH#wLf@i#`_g1}XO@(Kh-~4%eWBAC&2ym;+xQrRMj@9{L(MY>;o4BE%-Av@m_S$7NbC}ZcuO=Ur)IY4>EsJkZ+xOj|)o4|8%aN2j3(hE&yNHjM7r|D94uV}+WA~_k)HaraNdATbXD&~`!*U61oNr#hhj4veLQ@jK*CT1!IUneI1gl4t7n_NuDJN}wXqU5P^|=xvd}BQwE4mKS-;aFOmw20DHCY6Q!~>UGgtk=9m!R;U!`|{K0qU6??xlF9IJ?0L``+EG0)`NJuF2!?9@$pMqh^0THeV5CTknxv-8#;X(xq{6>sQBv7R!#9?KRn5%|zqIij7~cA|C8x#dupxPwISze1z8<fAR3n6KULW5fu1Nqv`Bcz1|EIFV)TKDT_-!u6M$Ljp6%NwK;<*Za6juA7cDk^Gh@wbyh8~dE@IXvD4aD3ZTWEi&gTO12g$DQ|8?WSVStc&L+nDH}F&yfm3j{7}bkW`x&hX<eRUD{nq&F%ineigU>PG-dn2Xg-zjx>2mLZT*0@;1|2^iki%ZD9@nigSHi;X-h$eHaB_{C&<#eFeXx4+K6U(FwgSKKFMpAiHBKd)*)nW3a|hm0t@MB^$ji*063bP~zeYZB^f9^CTZ!%v9#|~oQ|oi!uyauTLM?^mc8%+|VSV1D3n+Yl7%S(A@sWo5x=2^<-6hB-+`0FbfJ>b>|4#N#R?6vM4Ku5XHG$_}isXd%vH@7lm_i=Fzv@E93uQ5>R?qiNc(evO`pj;9|GGrM1%0R6`r}Yt*8*6cr3>;oKW%-gWu^l)b6-e(sFu|tycAYg(z#_pu(-YZQr7&fB&Ge*NN~US?8ak%cZ27X`~9yZ^~IH@r0fBVRJ@8CBaUKa=vtN8NXe@@$gfqMTI~%NDDaK>wNK7yGWkSqBN#$cy4tUeK98|DvG?<S*W4l&mrQkhM<e6r?-QT+Z-^xIHoTQxsej;+`D=hv+irB<^Oq|wk`^#lYxqA}Qenjc#6@QHnowoGFEzTN&Z}*Ban%D6G^CT(+#Rm*`2NDSox##P=oWjPzE8!B^Cf{`()&70q7xR+#`UXy!;Vi32D1nc7WkYc&wdysxqOa8dA#L|y=}c{aE`_Ha05?Qmcq6&3*R=;W_-&4uJMU0n02M@Pk5W1AwGADg~GizR%>h>2FG76JW|ofokVj{Q2p2%Q6a*i$FD!i*QvmFy+}|tRpg55c9JArZn{OKL!<Gj<u2*gQzqt>_?RSK6VKCPr*#LdKEQDARkmT-k9BMPb--vI?=t5qzbjnb8#i$D0}t_~O)feFMjnKAn}(0AbaN1j=n07a(n~}AGBnYxyAEx=A%<O<9COR6o4=47B0quY$XHe&s~s-S{rdZJJVD@$p}+Ty<Z{3kJRZl#`fM7u{D<9tiaI$P-q?))UA=)~o|R3OPV+Lk1P5;n4`;i#_h$=T5<m^KLrg)G+yix+zQ11e;})a3<5W|Dz1;<y=yr7c5J{_o?6FCAd5b3&hA%4d<&9G&RBqh6*P6fK;e%?%v3SqOC!B6PNu|6uB3!XIG+?#ab6x#+kE+k<r+XYPqT|cE59W&*!XNx*UU_~d(HBO6WRaU=;~~5&Ah@95i=WP#+&*rBWksmhmLmYQKQ>tG*hFXo;=G(%?KBdKa|`rD{v0Fh=I^!H_D6U--Nr$4GnAVP9^yLoCKhrn{wzKN=u}S4y4r4MXV|Sy>=+t>U8`R`N0Y+{YOO!v;5NN8qF*1e_G*#+8Skarjzw`gS@`|v)a<nCr8=9OzfaQMWf|Y4TfMqlnx^!b{<~1}zL>7k>9jUQY(GrG0KsoC`)RcD$L+qmO5mbK9Na&;wer!0T-VLEwsE<?LPW@S@4P%NTD+6(P$?vE8hTi!tDTnY_(^7)N7eG{T885~!6P;Ersd}=+t!DTKEAsK>0CZ&*krvt8&>8U2LtfiiZmEjJaKZxieL9C*Kl7oC8OHn7-ZEi$d4f`9#S!X=9Bdw$2}U$flc{4o9_-U{%azvO0)m&292=-4Hh4UWPeN4J%HaZf0`o(eZs{6A7e_{JS8ik{zl*AGI)I{<Me6-9{SiP^s{M7QFkjo;g<QFW3B5dOW-KM^2=#XygP*HnLO58YtLoZ9ahNg4m+%PussD(hn9ncZ{yqbZvC9!QR9US&&`Qt-)W#3Y;w`x4dRLm<>cc}>KwCjn0AqyN=V2J@Pfyz|0*~xZq&AKwfFC|NDQ@d+B<z{6k{i~CCus_<S0SuVQhU}SN0hATvFou;T@Ul;K6Y|@%xN^3!-&@yl&%`@GUCYi+HSWyLH3BvBTUWx(JQ3mo{Qi+sDUTQ1l&sP(8tI47XGT`hmwqWWagz&{$}=#_K(IYl!GUUrpe(c*&&twK#9j!3pjSs}z4v4qZeUr$kz9R+`GP0G#_Ck~>@zHV%sxG0iC^77e54cUF2}3lI~A$@ZNEhbya5hc>kGD+Mo}5DQuBb;ceH)^aKi7ei;wMQy!5ApU;iwY5O@`$bEB94ki}H{J7*=#Hg*Fyj(xvk+V|C9i)a?MeNMWv`m<>bX8KH%qOaz?`psHOQ$XU-jFJ@c{0Ts}b<M@m`~^|Afo5t&TFH^}Hn4D$!uK;KTz?;hJwEIMZJJ&}P?6A3tJ50~gt7Nx0XkI)M##&kHA&miZGN^EUgje<3o>mR~DbHKBu1+3eX*Z)p(EJ!!<FGtrEn$>w=neypNauXo1G<>va`O=Cpv++8_BlUE~}W%YW`lz`-jD(5OIW@w!0Sr!fxO^pudH9fp|@70D_m(4q`IBH86Gh6+Q-?-h@O`X<O>OEH$oG>ciPcwQk%ugI6mD+vGCwl6P4#1y9sr}o$zU<9vkVG<kP5laob*C-S4Wf5fPGv{4@uB~bm*>ZQiAZ(szMQ%7@zTUawy~_!5_1!N$ib9JDWGRU+<2-sSO4JsBd-N{5z@)97L1j2GO5%@uQ^;yC5PJf&hs6=I7F3&I;jS~O5gVT$lm&HFy%p?8(Q;w6)8q>2wXV7y#`Hh5Fn-UY!JEPSZ_V0ZM7genf4$TLZ;QPCw2Dsqsj?_`rG3$8Qj*<d8mG*U(#oLta;Yw?Rk6Hsm(W#>`foF15AJ8_G};*TdnEK>8X3I=EnfEZ$|(q!scYgEIQnzki*9P<UTLz{e#x{11ijFyZl;y#z!$7A-&_&xh^J)@HqMl{*?V<7E@+v;>MzXeXQ!&djR1R>D(Z@c^yXY;Oh7E7xFjFhA$g`Xp@?%P<F_n)zWORtCRIc;Z{C|XC-kLDh&_pL|ET(KfuYKUM2~Ka<&aD21BK>XTl;|8GqaRm#17~0^rYW^pFP&fbL)Zyj%)L)%Nq^FSHo*`-M2&j1MiwjfHU?ttVV-Z8J0Mr>2AV$ImB<4#Ar>->CK1Lu&2w5<MRf%W+u{+O0{o!>x=~iTmw?rw9-2Kr~_Bf|;Z>W!@y`DA9GVYIRrHp{F=~yvli`eXMxg(T}el<|+$p+>Lyw;(ko7-LS=b+5Qx9I7yd70ZIL6`oqTgv>3*<yAC=pGK9wb>SGkA*KySF`P%^<TvOrMeu;_H#!o+55H8OP17L@+wHcI*H?B`w2+$}20DA7RVtXBd;SIcirRcUN^>=O%P<npQeh=%sj=`odi}f`+1E)%$p{CUrEp>sb)}yXJGhY;Qtbx=j^TM{ZJGxu+i*LPNOm8Jn4i0~AW$@Ql)Lf9#2~kei*kAV_2t^+*x^nB?M#&*8O}S};mfgHjo6@ddwZXz<PSwYzUlw@V862ertX3B<A+5ha87{OfiPzQvus#IE=cba&o*xmF_E;z#c0{|D*b**W^In=KmB{;p0pBoci}unEPA&dG1~1=BHG|lWpMHJF<5lzb`%cqWja=J`&KRsQ7n*MGK7t}=30e1bd?Pexwn^|C1S36Djac`n>*_lj&Bd9f=2)~0s19iVypGlA>{9+P*<QAi)z$7%>z9(SvQT<ku0;=OsN<k<0~=jBh9m&}Xi)UF&ZraK#-Ho)L_M>42`@Vp0Gx6jxZ14ebAQht@O@=*Xt>?`zSbS-#O*aayG6UTLpMEt8C6BgTq(*9&aKOtWB<ByC)w6^Fr}w_<7wQl`zIutlTI#OZgK0r_^<_2?tJoSZG=(VZ_`Z2vR!%XYrR@wj+yKURQB=`IC)A7m1^gqp>?C{v_70QtK@+h3LmURM4zqPT=^sSgDv9<Y^;Hd*$(Ok+3EB%C2DK3`WeiML089}3)XN%@c4x%(_=flCmNm#mCHi-UdHDW=Qjy|*eh8*E2X=JR9nm0W)bHZ*7(*AmHf@Up}f|4fsqZw_T*yJM0(iFq^@V&tr@bMJAG_`Pxx#}Snqh$m>z?5y>r>zjPtXg?vL`~!tDmX>(HSX&Fkp=r^FM-#*W18VLHoxelpmbL@ew0Lpsf87n<5+7p_~RZTKR#m78$oK{4Eo#M%3AXP(mFu#CzmpLE<Hsfg_m&;Ot=c0*^no8QcF74HUhC26N@e&gozEEbl^X5S~Uqv*)YZMyURkfz!Yb$sHoifVjjtG#vvD@jqfud?6i;K|$}vSlgz92hL1pX*J7TUjppp>;xgg{5Dq##f+iy?cnj!TOzT+h}>AUznqWYw1b?fjwRf`b3Ddts8Q=?wR79Y@D6RKO?*_8}a0~uL>jTePdql)4nN#lZk1^!ZU1awg?FYTj+0fl;)1r6LTI*Q#-M(5#!kQR)2=iH*ij@fph1-ca5L-p7gpc9{pq}$H#k#^`7uE!wqo0{ShbT(rLA%-^aLm--`|P1Qjy!4m$G)JnNhV2}_LE82Xc;sS&|VE?@j4P)w+>68EIA*XmyIt^j|64M~*ec;UIZ`D=XAXNT+#>is?<h~wKt#BJp8=@FGyH1a;0j|YwKXL_EU0jG=fsQi>8S%=ThW#a;LOBQCj7>TN<!RGk6R60Gmit<CT1?v_A*%|c(vgIL|Ni_ZHh3Dn4T`*A6_*(#Tc6t?fhrNDo<wy84t)>eL)9&^9oJzrYIga*kh;2YL(sBWJAK;DiDeY#jTOE4nv7c{7wSSMNSr2nAiP*9(_dIsor(vNECYHdq^i3;-CWF+mAxMV7tsY)$mR$9&RE|<Ud*)9a<(THIInK=ctkM{&2rtxz1qpU95nT!hyjLe{7pn;Fot^p^QK0K>WrXCIC?AM-Jlz=&j1XK7QLB8@pLJe6u2H6l^Xbadpo3&)*o}sezcEJ~?lP3u|Gdg@SB~|&J`GJAt+>V{XpGWc|D}z4?isJaC;Hv-Dy_Atx{JxJQ_vHR8ppws2%_?p`qWE&H!q^Ly#W^xn$Gs$<^VlC9FJYsh2vp8+76grR_p1v-~Ugc8LIPoot_LO<kMcC&rRwS-AA+eX<<2vI2*CkR=t)!IEgH}rdH5F$(-JEmo~BVW))M*62eig?sgR{7tNixshw8%?6-ekg`xNvp;5gpA4m0BXUsdlgLmk%BCooZ;WhTb$}l**ci6PL8;$l7v)0!XCsC0No?B!_RMgltHS^jcI-lj`12~if&XJjB$F4)xyZUdB=go92v9r2*x8cfm?-4#lPjj39wvw$8jHngDRo`#1$ZD5d!PB%4RXHtBtGnm=_av6#JyVorE3J8l)9D$4Edt?wAHDI_q3(6|y}NrY8h>UCkq)KFcAUQ~;+|WRd(PCW&3O5CUJQ;GQ1`E^6sfs-RzSRu*JSDNU^~l<;4J9cu*p40^=Hs;*opg|IxNvQZsX_k?ljA*AdzrObT!1pH+5_BugzT?wPhwMzFEB~wEw!C*3#8AW|p4t2&1;xpc+4;rb2^w1HP1U<^zIVx5mrOUygn|V{%xz>SdkqVOmWN{x^8-2XiKRpEav8Opaj#c9u-a)SJIv1r_$PGG5HlUWa4NMs{O|^P-<<CjK6_?OrfI!$@1~mV45EKB)Tgxx}hw2l)14KxhiJ>o#}~<&6s-kk$Cf?}uLq28_z(!8a?_HbfHk!0`KX^iFY4KEHhZRFmBw`kMfvSfGR7!mqS*LR;8QyJj!JjyD%#WlnM0hOUYWa!zjVyKQq<bVtvZ_j2xkIL?j<c)USJrBZ1Un=x|Gh6~whPg}h%Gkwybc?WY>=iAo+((ed6_*mMlq89RrTtA@V`6nY)>C*=>&#>jN70o-)XmrDC4_4@JF7l_Pw!Gh^J2da)8jx;nxoZ~RNmWw9)opZwDL+kx`bu7%R-=q=RldOB8NKQmalW~mv=&Oe4_)E<RjS(n6boS_acv?_>{rNjN+sc>D#S*6UO~50w8Ori)~vo+ObqWnxb$RPNyfG^H_zj6^pu<nuJjb85}gxC*e#aZ!M-bvwLf-U`#mdc`XD?0>}Y?Yy)T2VQLEX>x_Rk<v_<0|Yc!#5=AVi<!PKSU)<|A!f60q6&GiJR1bQEtC-p4*`n@OK5`xT5h|8n*w<^h79b@kfe82uY>7wadwR3!)zQ!PKHicuh=s8YJ>%#<}2KeDmBD$|p5~#IldY{Fa{aVgWoqj(iCo?0`@;f=*GbKE_)%If~J>TL{C^m^pY$QD;%u>AG1x88T-)unidy1-(UaLFFK{j3g*paI9?h}5@{6QHX!zfzU_(+_~8S=2|Ijlt(hzz+kp^ab0U2HXP?~Z>M9b$IHDt!bYoO*EazWe4UqSqC9KhLN^a}0T_b1*@EDq^otMo~^qIl+3bq_eLT!QQK}*Wd0pMC4New$RC~{<v;`zcug*htN-l*(D5`Seq2YDq?M=(;oppxKr~z44bFf^@Fb$?f5`3QLQrD!Stt?)0_bY<sag1s**RS#eA4P-I@Ct>nl5!p_)oJ1P`z6pQpujfl!p*ppV>&OoIBlS83qa=VE0F7889Y@6LDerg5;L455Tmg5+?#=zPyWwtLQX;_FK6J8`TgR4p0Jj_}&u?|ROGZSnnQ;LhryMPzN2oIIxy4F4ig1I>KYu91m+Y6mgH<>~y#dvA3h_3`(Gdb--RhM_CEK{MpxfM(kqxUIvQ@g|vsZMPbtIZWF33A61!5zFEv^Qva*227-W8R;9~WkJ$de4~d4SKn(5M;Hf#-1Q9@<d){7Ss%THWHcGQessyd^VJeKucC4n-Uh!W3lE3L5g1;xwA;G3?}UQOlPly?IDNk*%hZ6*LiE>#*=;BXY#hb}ZeOPRqKF=w(d=}sD^(b-qKTd#euLW_irME2Jbj-^n!Me0I$(vIO(I}(HlGmKIneEIh2?g=7bzrzr+`gO9jTJ;XmVs=mcE;Vu`<y=pW&!ktI5BCb9;<Ew>iCT?#tV*-ZfUJ${wadJB)_>-1W*=NZ$Qz);-Wgp5MRms+73M@!Vauys3lNq&~0n?~3Vw&Bk`VhK~oAe`urh)#|vPS(lw2Za!1d2Q-lyWczlmEvNUzbe*+Ei$lG;*(&LXtbup3yK`8Y#>y*=hMkqT`24_9kH$QUY%<%C)|+L*M|_n<LR>%3hK>2*UGqU?wUSHub=fuUtLk2t2&}^V4)j=QeYCUk*Z04Qw@7Dg>U)D$#qr8$(+)NriS*mQ_q$jP=+m3xylR#675ZF|-dcke&(7OFOEv7&@3!mFJa(4I-_})$=12yxdba_iCKU_6?UCx-pJn$s8uEWB(@e68>T&TUe}=GgM%%+|$_)FGel!lBDt0+;;^L^}NB(+X8DJm#ZE_;5XLr$yodwwD2#ne6tM+Iwn<Pl)`&ZwQzhMo#6%hb8!=3ceoH?@;SY;?Oe*Oj*P;NlO%6f31X0QG9eM`8{WV6Q3PClXC5ob*$v(1gw?(X|~{h(K`xg|!IHX{D+&fEI_namVn5~E&P*XjBkCPKI_b?ba==xkU<XXKJ$T(;1AU(d$UU_kv$Q_TIoeMEvbtu{_w{BZC;Gw8Eib~g*0Dg422`kP8cwyu7AsWz#0T|3!^_n}wW8GeJdSF6=Pd7ku2v$t$bfmfw|6Np}|{f?VeaD(qnsRKaP6wSHuWU5~vTPZ1Ri-xz6ZH-YN<p%lDxfYln2QactbdLAUXt$a6FB#J~W8;VO@9Dvzg^^!j7;a%MR%eHgEJw%B)2G4KdJoHRq+w^{%{3mf?l(>lHS3t|S!1M#!g7)6^z@)0gEs$eTaVj3e(LEVWIl`9mz;r%CCv!_7u@^i;IwwRdsTXX=98a8_RehQH-0ne06d$o%^{$=s0mvPhXR_>PJZiQO&_5r_2ix9rd9(}r*%I^zuUqzg6w;#A#fTM1hIXf(eypJp4BV*`)Vzqa)|4lrv0;LC9l6jwD4=i3Eq&Tj}Ab}q-1RCghz^9y8gFgWz=cWZHx~2=6aRKt6=gZQ?v;VCixvY$tn;185&4F(-hmA>652?C8w(srOv!wOf>1D#_twT3T4biq+ylsUfRx}NYyk8yuiyvdOWE@ziYjzVL6T#Vwg4V19PYP!Yw;4-XjmbJQT2WF*Hl7UaJVv&0=XcsXvASKZoz>`1Q+PLH6R{2<2OhnkfMhP#o=w+mM}Jz?st{2lv~i`_7sKvA=oCQ8cNpy74OptP0rN!3%J^Bpq-(z?&;=zNEtvvHIZY&w1WyLGXvac;TC}(X6w^6(mZ(UwL(MXjjJ9<{p^v0Jom``K<Y52Q!UnN~UFJEiX)z9WvYC<)p?HU%Cg5TC3c9jCehrO}u0?*fCT(vfuZu_5%@t-l>0-)bl?|yj^XW1bknx#AjlGKCB+U!F@)QBja7%4e*3F?RT*7z@+7K2U31lTf9ZDQ>-Ghs!f<*Ya}(;3;*mb(Q&Le2*&N><9d4bZg01aKqL-VN;JPfRe>$psUm0fM)K<Ejpw+_J{av5o|3+YH{?7YPWwAxMiwYjI~^~0cwf<^?0T!Ige$l@U(^tA*GR&z4ZwJclE3!*rm<7zDU>bx`|Hh@HNik9+L+?1`l+`&uYrLCpL&=1G`IxDQM#7CT&vyI?CZc9A6QZ^D}8r+coxDo%<XI}H$k{u-R5&MT;6Ii-<JgKX~*q9S?v@Dpp{+_(A<rcqv(7x+*wqa%7~k>yu=&Y8maxd{ZTG(hlD%~`TX@jHis4wQm@AGUUN>Qeph3mwNl67YBKZ1OC6YR8@%rVeU%Z=SFBZV;>7hwzbS7DC-uiu-TLTkH|Ekg(PO=Rp4g|cdAcVi7oqD<)#ORnNLCI5rS0L$bc_8*C~l8Q;f`+*F17dZw45?{x+!0=4!=K6e^%>$@MdLqAJWU?zh}OjmiI}tt_EJ62?v*kG5D$dSaOnN3ZK3^`}eix?4q&t{X+})(k;DPmty#rK7o#{X{tj{tCLn(A04*Tb+;!LVP6Xy&3OT1XxO}-h8b+vdgL{Vb+72`;83g{_xTy^*N^Rg|IYeCr-MOAQ2Ud+ldy%>ctf{B=5KryQ<DlGrDnbVG5+l2cf6yOT65;_<b}{!v+I);?(0VId$&a_%0s+Cq^}GV0SH3@I<}7B4CP8yRT5q-^j70=?0D*l`V>qwnmp?t(LezB(DBT6kTm%{j=A!g++*??$H!;r6UNdHfXY8%DN*YU>_w#7A`<}J3a?pxJG(XXP3s0UKlNVyH&6a_=kAeu*mIqBB~G<yME2!H^Gpk*x{%4?LbsU<>uZx-?CxHOPPf_1+7$l2czKL^gKxWmO&8NkS^e}6eT9mFqT#anq;o*3Jth7CK(#Ak{^90Eh3j<;|K9r*8+?2`?<G8s50*lZ0UDTH-+p%f620Az8eh*lZypQQ887ElJY+N%T9>KxOsq9EBbK{X)~8eRrpFtkUBAT4Roh09+cs1~P-JI-wxbQU`>l4x`!HJ_)>nZqYAK`M2DOer6YJis!?mc~0?(C*845e=H}9w_c#;k$4DxF$W1+9xc#!p*_6?t}__7z*#}fCp9sh9^$KzFJ`3G0M$_zLz>f14Y`*EgkNOSnQ`9+kF6DEtz8vUF8(C51&##GOZjW}yf%Ko<Q((9<UYJ9^X3Vml68hm-{nqRyKgo2_<tG&E0tppxd7Wy}l<nf>K*v*?i8?0#~CyB<0tGl&s&H3ZT3+|2d^T<8%A}95{*ID$xgx70r@YPl*)vPrrCt*v`ccAYC%f)DQID&!ftT)8-2fpB9I~weDD(mfqLOmEZ>hM7udVL*hJFgfdG(wGccNg;U9saD#D?9k%P(CT^8g_40uh)p_78+Sl_;8{jDBOJ$lX3S+ZuA(arjzZk8ZNz;(xD%FFW_+Z&{33p);qF8VP6rodv@Qjrd9L5*?R}D^;UIwo*#+5-=(-<Wb;N@m8Q(=msj6hug_@>6r9y+>Yi5^{3G=qH8Pz&SJD}DZZL1piz>Rv+l;i#`-mR22|LKQ>#xj9w`Z~_YE*txiT=w#=X9?Uks1OdU74RWkYj^JR!0ei=6PIR@V7>em#cAwKN!tP+-jJr`Ipeaug5%k_5GTT_v&wRUQd3Gj>t*v)t`968^3KHL4hjPJcu_FZX)%m0kl+r6Vy)nYA#++Ba@q;ZX>@%!(`9f-F~F@&i5Y9rEC-Z72SUG;f1XtX4A#Y8PwThOF`s1w3&iKhLWaIYA4m@u2|fwI`lieI>vPgT@DTh(PX1KAE@b|-1db4!i-msW57(=Zd+e@F?3EDe5m$|*jT>~I<?i58sw-qV%^Rf%Lq?e(16`S&qP;`WvJELqbeD!^@wPvuTFca!b}?l#=te6^S5xw;PVn2Sijk}@r`S(E0WH9<_KNpf6VPyyYSaNq{_bLWq%U66orWn)AdM<F&lx|woYo5HT>-f@EY`AE0&1jT6u0>DniGI&hx*ZzSSO_>sET|pCgi{cIoXCY^-*e(~KSy<?v&j@p>)}XGQZXr46&!LxfeW_SbUact!C?X7(7EdGRqz>NCc7r6v&f)#~u+cTHmY3T^$>+i#2?8i-Eyl0d4>4~&^S*z`$kKD`o{W;AMpM$sNc_oMR(#~G3m0~_Ekb^0fc0DVWeb0EbZq078i0po%_?osJpg!Yo%gL5q*UP2;QCI)|#CQoJZh565AyrVfsMs|}hNgsd5;t5PXLFqYN9bjL4o1~FY5$O&PGt@=;9!m7f2ucl`b=EgGM2M;I^q&6pT7R)8uSkpOY_%iKE!Nv0$Jo~a^<Jz5?bBWszO-yY=h=rZI$Z`v_ze41j#qYL3rR!08s4zRohHtrcf9nchAri|H>BTH9NE-<b;E=2*HtcxBV&YF<O{G<<GCYbdu4?4K`|RmEE<qGZrW<UQO1AHDF2rN4SeBlfBvjca@33~^%0?AjeN6UShZTx3Q4^mED%mkqu0DM7mwd!RL+5=5#&<m896}qePI76*K4}nQDAU7j)Cs6h|5EVry{|T>tEP0_C3l8ylL}X8vUL!&3z}&VsQCZor-X#x$V)`+-yyuhA&Mi`Lw!UirTp}TeTe=*OrHw{eWZ8vIdvy4mAj*ntgjmd!7KwtzIo=dnJ)$^82bo%!{2v&$v?UaeCJrbl_h2GG_bhNU2pGcYJWY?w(G$316>w&)b!EK4%o+I9zcD3!g0Q&aSi9AG;#$KjFrTfAPOzm-zs3_cdE`TdyV~kx{9jRVM@CVdn=1v(>&5^j`y~`({R!@C&pp<mV4hbE|V_$eknNiH{|9HS77?rL}H8$ShiIq;sb|EMLn6Y#Ulg9dVq2=FGkuj5gxA01h(sj>ikVSxmv8ZAA{$9d&^KbqWF74aIHAtlvuW$BFGL-#z=m@HNc2-5akX+q&(jZ{|F+lD?~vYk`G+FWQ<I&Jtf(I~FCzTcpkjbzaRH<FPGFQ>NDG2J9R0hQ%VC{2iptWAgl4KB4ooR?g_QnjWR#)fzt0iHbkG(|g0#lHRkP<Xf5UP+=B2gkn;x{5My>&Y?3tdt5{R1{?MDl%Vrug<`};?f4(_`4W?9yuY-G0O=jZL9v$F1hik$4tRpr+&O44C3P-{IWx;V9zKw{xPm0AaqhyS*=A6_u79@BGGge@Rrqb0pu4RWAF1z~Wx?HurXKXs4;`;AedjZ~BEBRPgx9hGT$a7vbRT|JzC5^Ec-l(+I?4OWK%^hG#Mukp70fDXz3jiH9%AqDVx+!pYCS15PL!FEc;jB&QMD>f$-$7e{Qx6%;{}f?Jg8~YK&c>G^jCJ&S71?u>t4@#v~B|x6WK5)YR|OeKSr+lkBI)hajo*P1H0)~_iKIKd*_aS+(B=TzTB%QGAycfnQ>_0`qNAHpiY=(&;QA*I<>F$#Q-dwHi=c)^f;<mCzQ%Zw9x37>+WQy(09(f>hgLZw7EW^`zcJGClcf}8x05k$p#nOJNC|ayd3*TC$w?D>u?G^yj}8E_eGfR+QdJ>AGF3G^b0*fd=G}z@`c|X>~!3q#V0TxV?_t-oJxyIO@DrumEj`t3P<KCbpH+P!IXi0L=w551Ww6DVb>}~{v}5#6dv}E1Z#Y_>ywD8!y9}00OOm^GCitMB5rsnF3E;FoY10x*v&dPjppQI==OFJXHmONe>J1@d$B(e1!cmI6MJ`PFuW|2fjDHdT~Yk5KeuLuYjCwJUfE<sMsU&X{w8a6wVZ+T;>k7{g9Ih|FJ*<sbbJ$f3A4WQXG{CTCOp$_tH2j=e_JIg0r<JpB~)FTL-)Ez|19?|ixEzHfK3Mi8nk|Kj67m!wTN5vehAorwF1VC(Hp7Q=C-<&1hW5`RM%|xx!Ang?^&ZV89E!FUAeD@s68P5Qbes1HNS=q=PUWtf1_Ko{g~_-QIt}#f@p&ey|fDVN@VLE@HYpVYsXN<rB#C3qOE#m43^}OX*{=!;69te{$_N)`6j6d!>-nT^5I?uZ(gqv%bA|2mHud2qqkIhhfw!Z*bhHISqr0GI^P>LV$wRP;MhrqwPqfbr$)G3In{5@xHqYa=|HO##-^9rHFY?zh2UG9lar8@w=u_h>u$SijlAm6UT)ZDYutOW&y;0W>-zLm?OR>=+@3xQLXPj(6Or$4%O^V?s3CalUsp-P8sX`^xs6bRc9`RN)_5$>$Ku+wH8SL?lv;hQ_uI*203STOSN{NsRgTDa1H%T3&!q{)RQ1}T`=UfZ2TV+k(@JSX;BHvAHryJI!}K&U7Y2#ZJctKvXUmXYZ89bE_vN6%uAtQ`=}fXZbs#)IE2r)4hCv|UBlN2YII|ZGRk@<UA!RTJuin++h<946`=_9lGO7ISffc-cPY2$xIsf#UN|o~7tC1W7%>J4!YQaEy5Vcl*J!FYIY(cmHy4gFv0N8lJ3p)AlPmsUBmX?s}$Q9H>t}FYiK{3zE>rVLIv(h79*L=JSXNy-2dnmJk_&4!ld~Z1O1WnVR{_ta}b@ivSdob%ny%A?FPlvK+bnNYSkx3o)rh?TMNL^;gHJ|>h8yFt(0kc{HR@s%_;sJH6A%+Z;^85G6m(BhoZa4bf`Ri(G$3wkDeS-5!<u-b6I6JGN=bypNZ2L}@`bz5k)MmxUEPpQbPC7LhKh`>aESGV*>WTA(9&}TjusTDktMT(tP|x=(A%B+{IBM~uYaB4m#<97rI|l8ykoSKSooVxOU=)Wxiy@PY8PdW?MY6Xd)eD3cQk2r7`s~mDdDn}YEa#lxb<J?!#5dcso4PodFYJ6&nCskk>+=D0lB;`ea%R`I-{uy+5>=Ocy{|_#iq?i*CUn^!RGYTzl^J{Pt+xARXFq;QruzSJv1HDXqe*=)$TAjfHaBmw$MQGC_ovLJpI>>6nOHwW0{?1yCa$<goMPZaJi_x~a7Bb|v{QnXe_Wh>yBQz;!aA`dr2wM?+;;}#d$IR68=Jqjw~g;rE%Amiz6|QXx|z790`;OFuO8^*66vCF@3HNP^h9m9uyd5xho|f1{9b!B1p{n~{{5Y@+S63G&R!2{3yCN%V^2S(M7XqGuS|*<$Q8&*BKvuo+6(P4_%(NxvbwQ@T{beQV<CZdW~Z%-I$-zevElJDO$4)5AM~e9S#A#+RpCql@e1|muOEfY&KlOb3^88)9%AMCC{-@>JNRTdsjup!*H=yIbbt;gK2EnMwKrK=4_p(0RA5ud7c_p?q5Lr%jpjMAS91L`-JmAZyq`1j%8lyF++o_+HYLAzupq1|<*3R*TE9_no^uP0&EC*fDxO&P)@3H2DWSq6veeu^2el)3boUGU+aK6+mX7<=5BdaLiGM=t7vO!cTtF(zD~Qu}WkV?lXiLM12<I9SgT1{&<KeOh?@QnS0i!x~`c~XE>iF_DtHNwIa3-|@n&+KO=l!^^M^yAoZ`;&kI`zKVm^$@KDO=xJ)C~?Y0Y8!=Hx^%7StLESkG8a-s=^g`-Iy>TEBY%5N)fFI8*wt9JUmt>I-5$8eD{56epM^O{-4G_C!c&FM=V@DtZOojKO2jtw_IuXrt7mu5ktZJ>=RX|Fgq`|CpHU?AI!V}1MqS+zn(ExExoNT^dOHculUNZPL`?o$m7EcatX^6uZGCb*cnW3V!YfBELN%+ehy!S?n^EoYj0h7YPE^l>&v?>X=BnRBe&V_wlDzqEv()sNZ+h#)GdTzv$?)<+I=A!b7ngFOK<c|U0+&W)67@$I#}KgmYX$uZe`l`Gs0g~NbAZ&2x-@G?-o+dYcoS(#<ie7>Vr>YdP=fYmz~tl=DZ1(7g1WN$K72&BS|+BKMJ_H!C*YRO|yygNr62R2f^;rcF%Cenxlh!U+y~;Fz>dWtJb<3#mALnMY(HzlaSPJlRHO8?+`ZO(*?Xo@wR2_rBU3pmUd_W%678q^V8m}hHmw=xFtta^drXCe+_Rmo1MQm7pS<*YEC1b6R5*Oo)W3nTT}5ewu(ppMf_}=kH)9o$6JxTYV(<$kJbQ5>gP1y!CiGD^VODnOGaQKeo(reHx?09C96B^i8O=GD0c!MMb=edek8>YV79h>vp4wlalgMXb$(=IN0Riqv%1GL_tqkB#C#P}oC>oRhJNXfM+t(6R{O*ER~>A40KC1itk1?f_PJ;ttTqur1I*Nu#$a0RmHK=NDPf{!1Ozy!-Ow5nlh~Vswh~{jE%Ce99vpy3tHq&)!fKgqN285wG^AlCMrKd!s|Wa%+i0DOS__}X6q<m#OU{C(rAWjQ*anl=&g^Ql(NNKXLPU6lxHeXeU1L1cQE8B4$HGgmwlBzv(%IraiO1i%P<zCS_H;Mr_x;K{eNI|gVshpqk?i{hYlnq(aE2}W9`35jx(2?An8lwKwb=sk4JG*g-y8=;>}n>|zV?^eA<3ejW(<TmC$5WPA?Eknd+o|LK3(g6-yea(I)DU~LV*`$<Uqeo4dyv8eO&ZEm!Y2nW;hL3U2;R3Z(*Zmji{z$QDtkroZGq8YN*Dht3XGffY*F)bYu3*IlfB~shI68Zu3TJg_)7aY^F%45frmW!cO_OsOz;E6IwhEuq!BgZ7R5VlH(wO{#3`nQdbzqSretMt?PGX=~t(j<kDKNo`e2ph~DyB>opu@RD5hR>%YvZIE!9A6~oPn`gMNqdE*d+YbR*x)k&k4F}Lph$f?C>=tIp8vM~|pZ35dy;qI?bNxfg=eIlX1w0%4=pF;4O7JNMMEAeO?eOp1NUKYmm+F$4wyfFfq)<ZRn*=9DlvY8E)n^Ns|^8I6Xoh-J!b~Nf1*Hwc@d$Ifh4y&%3^7q{Owc-K8;FNDD`ik}NWOcW0={(Kull_JfcI%eqs(OB}+~+EIK9+s`K5UjLd5_PGIrPeP+N?aGnl@RQWh%ZDMz&u-!9tr~1psf#!=9eNPVFn-k4+pVam*YzP^-{ZI`c=CL_sEXsE@SDIUd=~sr(M^V3cIf$LKoiP2u$gsV>!fzPnAdn8qZ+q1a;3H0Uu`tY~f8)vENnZsJ>F&I8D0!QPOi*(@!|ig%w^75)*;@n_q1gU)~sdXO!g#@qwl2g}Hg&G*vHsOj{bXS3#}iRebPeoT46P)hW(DW8*3HrrpdN{a4@_w^$zlYU<~Z=<xlR^|<IIcDpZ_4X@syrQ53;O`SHH;bLIo<Ck?i#Rr&=OQ<zuR~L4rlc)kKY0%<-IN?C(HaH3B5c%W0W9-=&pdp*ttjSLcTYIInm8M2;{dpMU2J@}u-;&4&%xOVLXMm;^XD0U-t@NEVhGjwZESHL;Ohl|C#f)rX7?)ndPKGm1@fm_rzE5~n<p$VYnYaoR$3)~*W(Vs2bw^5n#?iF^m;yqB}@!^zBZK)xu;gPj^BGae+Hb)@6Vu)ossLhvU$L-3t`Cfx3prENP9Z=<*S5@-c{g=K~>OBqY&i-)7L6zewF;>CnNVR(wZz=+rs7GA9f<==c0p+=L7TeCtl(A18qm#^7%7jNm|uLG@FBgq6nBdf8>K~lF~Uwq*P0KZPs3cpO#lt*kt)3VIS*N3wEj<b<}bUw1p(EeYNfalhnZbj)y6`!F%fx**83&o{Y5i1@qU$2kU-AdisyB#tr!Gi5@94_sc$~eYM)G{hTY795Y?v8+$&4EB9I4haNNesD##oCET14JhWd3zZ2pouQl7KRK&k4tsJ7o<{>PfsF3ZxTyvsJ3s5q5>)|uFUA>mG8hTKdU2KcGJUV$2+RX#J5Pe=IHpb8$`yi{&nO>1Ahtoit=a|-=-`@NbL^AZfcs(Y&-){BH;OMei58-vi*bnK)VJv~O{E0&izM&*?+Zg3DxRWGn@~wB9W-7$DwtU&0?^Y`BwJZ7|hW&^HUn{qY57Us|@)1^hSV^^3Sj-~a2l7HM&WGUysNr4(jz=_DUa$MB(oxTD>C{yKwoGRaOpuKYQ#dsQ^il_@dHh(QG6FU{tGp?dkM+^ApSbkF4RTnWWlSz=_4VGTNG|!M=izR>tZMeNdx*uPsc)+LWG6a;-l4BaCgRh+yl-6<HScvYMtCT|qn-56#onv7bQ(Q{f6l^aQY9FI?>A$(q<hJvc{gUvjq@C1OO$_({-%?-%^ne@Rs83C*2KDx%c*m$y87@)SVFtL;`M>|j6EGX|L{inKI|%j*bV*Bwy8ArDNHe14Vf?9HB=i8l)jCvhljP&4oXCB68fx$wM14g;vQYQUxnpcH_L6XU$qe;o!yn_QUA!bWMWXy_w_XF*V6U!PvE3uV#aS)g8CJHe9!i#uh|FdvDCNeLL@uio<hUg2hH}$iMt@53qPzy>R~5R%xqhHEP%u0u6<&%JV_=6fS$)2h#a!^jp0YNDxa_*R(9iUe-?BNs4n!^GpoaFDTq}kaXA@Q;T~)P3kRL0cq#Wr991&G{d|0Mr2*8aE!}!tM;+r%t#oMHO$Kwt-EpVgywj6g$M;vLA5Wn8zBo(dz~@}fA6fP>txi@OV}8Cro?TICtNG|%COE?@K7Yz~7OE1)-(2r>kD-BA`tLBeKJHq`{sw;TJT^XAn9h}qJdkqzMtK+>1ZTP&cV@rpgOh;83_j7P#WxIFjkh^$WAto7EWsBuq=RfuUZJaVe`F8Ooc$iCj%a(RUo?)r&jsY4(bbxUoNIBXi5l&X$9_{4BKkAQ7kVY8H_DqMvaYAELBEnrr_GL0yiZZSVWnfUxoqmj6wvDTWPf57N&g`z+RTgJ4|#bygwHr>(GB}_gP)V`Zv?$LrS}?_`?%|Mr;O#-7-hYCi#DoaHN~CtS7yKc#^kMg_~XmH&SxXz55s@k+W<elo7;`loP2F9-!9ckJ--<#?bJvdDxbiC;xxHkoh{$J5bcdaIL!8vN;W3)YfXR0Rs6bzjwkFqx*NHQ&vM9?8guzNvUg9dQ9bs(F=GCv#nS6KiCj`tJ#Ae9KKm_a%0%l<W}G!yu<c=gM!xp_mSlFF#^Y4|8P`agO`*$~P{LtXwRDW)Dx!o92VrpVcUtv$y9?gXsagX#^^(JIWz(9L5DKYDPzT$Dp8?#g{lR8Ey`8t4-ziMnj=ZbF7Ef|1c%dN1IDF&)Gel!KFn2A#+T^<3t9f}e?;GWnFNB}fT1aX{;={oia!rXe#=9jjaj6}9JH?0NqggVKZj-(+landVlgbgTQzW7vMAs6S=Va@}oztB&_B?0sXV7Y|wW?r~VI!!VI(1{J`O)w68ByS1U44F(zn*;jyjv;^z2Wq(Bglb?nknK^&GpQgPv2d*i&}j|HanXFXp?poBiohlU`-D}a>$hGV>ux+m$=D;Qy*AG-Q#}Qp7Nmd?p&Vw@iKhx6|lE+-{1Sz0)Ig6j+SFaG?W;j`V=<o?3wQO>0!B&G=v>{G0zu?ns{z7D^9%$4hU^Y=cHplL3?&%=8Ue#N9|)1Rk31fwUu{iSOeGRUu}cmYi!I<lXT+$s5waZxJDAq9!yua)dO}oc9j#dyggAX>haY|hljt`O<Hp4sCN0f$f+;)X?d+v*a#~6A+`P}?GcmqvAj(D6MmNNQ3zMdtCBEeH+~-Cy%sE*l}*tdEyShVy}&E~P%9^Us8VlVF89klz1}@`a<7)D9Xv?-H<{`i5pX|v2Qc>XeRZ0#fYK{Z&Dmo8LV&dD@6Hbqb~vvHj%%-vBGtyTLYJ@i1{;omM@UW@r<G#XdQGtY(NP|Ou=53<k74OO5H+4dZ`vd}P6T8f@j>Wie3tH)?@l)A`s>!WU2Qk&wa@Kl+>l$BLXmDk<K<MnIv4S}X-&t@VGZ$!C7++EIPI8krfP)9Ji#0s|CY#Yx=u&<F$#ty3bjwrV3~eLOsD2eTJMv3YnbLN?EhxJ>%&zWNG84bR!}@yw^P-q$7d`s3yG#OBlb?q4S47r9-`k1jmmrb_tmo8!F^2GYco)b5ggmBsn(x++>1|58Mnyg9m@S9BVZ@*C~p9+Zsg8zCT=UU%kkZo$6m<Vt9R)Vjs8qZ2Oo7BRe`&AIwUT^i`#V5&HYX~Jx>k)(JR+qVC4{ILe9ZKVuuxMoDC=QJ>Ke#RAU%e+)!dWCYo1$@5M6LQWz|sHyYmbjZl!BoPM7N{i^L4R<S*iM%p)DJq^Dc7i;7Cc`vw;03CzYv-k5J*`YuvZTamusuNbk9SN!EBLJ>wJZf*xw{>?wtmX}TeIEXf$g+O*biJ1+)jc85^!!L~apAf@U{Sh1_lM(E9z(;{+hGv;{;K)E^LsaYg@hj6dtuV9(*6lp?tAB5;P<39Ho@2Te8N4VgI=Xfh20LGoty^lpG$D&?IU`dkKBnk9W1L7+){^w*fef#UU^7S$By7^&}vfRi+^;66B5^-IAu(GS#=6zk2d;J)xq9VXU<K3cHlEuTpEW>?Rtc|Ad;x$ps`pChls3pCTl<_+G(+EB@(;!w(0TJ+3A;51O)I1<d|#CCkt*ST<q7)3<o0&Tfwk6-KJkbF&_G6U7DLU-)tkHen&3{sf!SwhS{VmLwlp!sY`t+wzF@&!k9D7QU5ZyAszvm&2e|*Q+17*NhMEGf1L(H3w#t&Kjc2H)=#$qa9j;MJz0<;#X;r1KJ+fu*dN>RzLwN#&^9mQ65$0FH*bqYJo+4#XV5#(2=4_B&Wm4r<SDYCaFgxwfaPmUY?7VNN{ggEx_K=xWPlST#I}vYz1^Mo>vrL_-6L5*{wo|}zs72M+snbzzt?ZRURQKW8&5Cc+1}dyUxglRSIW#!sN}#ds@Tw<(UyFa@@oTC9CGQ@o^LBv%xJ0;M`Zz%@a|7@{_f&Yp=K6bH0Xi0b;rARzrUe@7M$KtseysQ+TT28tDr{=H(!Np&49p#7wR}BzlV8qxne&Rkf@xivz2pSrthw(USG8*+n5R!dAk6<I;}6wY_X?!yq4%L+r5gfaw(1EJ^_ZS(HJRPJ7N+p_pRl-4RDL>ggK`^*!~MD;e7u!*oMB`M#n2Ud3Kfp5{{KN&O8EJ24Cc>3!|5=fF?ax&EsMW;m$6%@Lbw-A8zBNz6RwXtVmc*dRJ=WU$u>G3}Lw7hVz&FyI>DsJ)IP?^o-jx7G7wC{nXI^Z=)-p?GN%7*ZM*(XWZDm9V6&vTz-?_fpS%aF<zZ|X>J$FczP<%{Z05b%u%}$o|hHDz>pNOh71wn8N6j4`KgTzzHR5x=h1%XCtN4vYPV12aBJ*pyPoNKb3xve2Xi8?Fxb@1KAYRe%D#_XJ+B9j5xYKi$iQ#doABEHbTr1R^niD~W2$et<JN_WFkkjL^8tkV+R-`M>}}3TlmGkmR$q<|%y9QPy<Dvoz|YYxMXgI(uPn^Z<%&!xN_q!*zPsfF`&w<Uh4nVDc4NR_MZbSuk4gNwIgD=4CT>ot_F`iZ7huy}HB1TmTq$9|f}Qh|nq>CLJ<<6|7}S1edN+C%V?mnB{ACm~PkwYvDp~QeltxkU%u+5Zs{CLV24$K{wA*1>O1xZNaa$f*(D%f(+<ePy|Cb)Fpn{21i`pLfi1XoNC20pz5k$y+vtZisMnoP^o@Hk1BY;>h1k<McW1Sz%3M~@k?c$3ENCDY#QoTsM`@AK1-MH2f)UX*LPQ#L>>E8*QABf#G{xW`8T~iHKZ8#8WyazJ*ITLr~^SY!I-4)0IW)rBR=Z)-qVw)&iW!q|0(azv7sr`W=&CZ5b@7K}e=dfV4PfTMLEQkHHEwc@((!J0vrL{g2v)i$R&b!H#-iVv(wK6+Hz3aVpkzLcC_m-1h(xPCRXlQc8O_$-9IDaczz0X^Na1+}(wz^>_+{%~Gt+K;W+YDO0U+Xbvkm5eZ?o|C`{xOtt;k0Es4c0w8jI=Eh3l-Z;bRTNROV=c7@_Z3|;%9(%G4FD?%f@@EF!n+Y8^!i0H#pqq=Kcse2}1gQzOM9l^c%cXP`m2mOYhXaaa(rB$1j(GaRNh5^#_8<E5<dgx|gjU*bslTSC_~9{{|=P)xEP)u8R$BnVuW-?Y)VbJBi*@TAft%p8%Y;_Dq&B{O+4wN710h<<KoeplW`gzx;;SA2(FvK%ulA5<uT?mCNB^^!k)t4r-+7pk}b_q7GIrf4BN|%rEA5gASt(J<}?eO<N9ja_h?ye@uua3inuU6S@(m`JvM|iH+uu^)*q_&KE+EkW;w-LCXA`9ZCCaFF5gTwlbYOAH?6`*~C0;xr7bK;tG;!mgAqvJ7NZAtFC#TKJ{|Cx{+mbVAp28@@X4lwn?2DbyAzqO)D?Z%x0IzMOzv}YiP<_uh!|gxEG}rXjDBvZwAVfK)BzuC;F972eE$OEg&wf*LoVV6RIyDxW9l}FTcXUglk{5=-~7|YrK*BJL=&UzF)KDQ?Q;GFZz72lc4oj(4u`w3(K2Wt^TyO;kvvLB=dWF9vp9mignpctK%EI$*e!EZHtnMV|GuP(zRpwBtjCEbQ%)1t2VBsE<Auc&)a&uTy2Y;R>btB4=VoLt95|E0;^|1W7~4+!R35kDL3>n&!t_Z{oaGu8+8O3@P~cZougn@P{bXSGE4C{;?^!3B|@rW+yValw>VD(f&S1;VUJ)$+L88h58s628f&w6vz(*e0?gb+v4NA{1=1-roMmDuA%%Cfd6b=X@8@>ol@Tc(J84rawCPVQ2Q4;H%?g~4&1lp)%0x3r^h$lW2@#7p&iC?ls@$yZ=9z1V=?erA?O;PMM=q9vL3N2$?1<zmtP*)^3wfLem13tW*~P1H&GCsa284mI?K(E%s}1{4Wu2C|+^E@QAJz~3QRiEhXXBeVmh$lG_HVqK36J^J^<z^Ulg$`jy8W8@=vR$B-JY5IVBydb23mwIUKHB+ydGEkSnCT!(d7LXk1Xca>_yE#Te+Y7^0XdtM#F-)Xp8T?DcW^cK=boAXa}&mSll`GKJ@5=0MmiERL9OS1g>?d-mhL(e;QXkehFa1XxVjRGiKQe=ZOpU`+R=#Z-@0>ZLM7@gY)eVvHOSI_;Dz9hC;k-e~Ynu&vF)HtLZ))Enw{F=;rn1<h{7>wvGF8xJwb_rjldHCt4SZKlq0CjNXW(eqM@ww10uC?wmb|_v2?a0p0!6EdH2azqx0x%1{geayBO9+I}=0JUBXTx%<jQ$0`?c7VtG2UYnUb4>m%?OqT??f-0*6KA2Y)$;(~!*v|C+yj>UoWd{rS!Y%8cCAd5A=ce7`S#UksF=~4r{!zP?2CJ?6<?=f@A)cLTo}bon>;78&#(R0o#M-b={AzG(Wz_#Bo}nlXsH~=2^MjWpkIQ9rY*j0YRat`vx8-*$?HlMo^KRoSKY=9+OnUZ$uHVi*EzB%n_x{|F&iLMrs<wnY$H||3{X(NTSKadlI<<KKuiO3XE38JhFdy&cMx`}Lz|nlGv+XZ1pZmQg{yj^-H5M^U=jFS>xB?XlS27iObuvHmEZbx)v^(0w^k5co&ig{8vOgw|EHTQvLAo9FyNy|WR6QEKO4Ax-%>!Rah8@N4z%I0Yv9f-oMZ@auhW5^}cW-N)@~4AFVCO9Yk@$^Vk?dB*-v`^uA8$XfvQnC|z84;sm4iiWHWbbePCIsgxh#V5^uarW+TKGQw0)r(^TcKklY?o}%@pV!R#b+son!sWyx=Lvnos8`1ZrGPg;Tq=Aj`p2Jhj3jm(I#c>9#g<IHy@?BFow$1gpp`mI1L&9bD}TlDs!NYFB=Brqp0?ZB7O&CZ!M?<Dm-TLho^OHt(0ubW&%>K$V^C7Xe>ZxLrPCmEIF*i0~BSG)`XhK_-}-TR~~PMw`n*@$@G*`c7Uq`RpPWGN}|TQW<XsMEk_zg^UI$d#gkEdLPaVr?&2m{*vY0-Ux$vhqzo?Jr69bY1aNwZphW|^+su?I-*V0-f@QBUVP8FvdZq;>Gd4KE{K-2u3|`26$C4^^!-QJ1?Tthss5E`7y6<ijSp(*_l_h$eYZF(VdulxvexTm_;M@f+r3PQh8$g*Bl2P6P;<ObGJ&WL9Q-c#C49#D){0jt#y3jY{Bx*sQ$<#ftvdE2`vva7-{UOvaf4*}`~9AXuJvnF8Zio1^wA|TYUcd|B~k-FUcBdb7|$D)e7(ZBdW;VYwmNiNqnpvpi6n^ya$YoxH4GH~@Z0&VgJ-8)>%-YG`4Hn?6wAFX=q+kz-sItNm4_$WqS3$T2Ym9{u04eTV3ix@#yG9aAL01QRVMXUcIb6RO{JJe9UwSl>^$G3z#>TRr|#WemL18pm|Y8_W>^_pH_jv0$hp$visSdC7FW`56`k;{`QOT?ULASQq}sdAmMm5Kosz{vkKtChjV$qPhgD3Yikbs#?xVmI_qRzsp!{zyTn&Z`HQMO-BkGmrFrj1<=GhU}0VmI$bM2nFv)}t0`v-jx^WxMIUZ{H+G71i(T1VyM5C(Ak%y;f1NK2G76TR*m=UN8<u+K@LN&cPVUWZS%AM~~#_PDtA#PRHdI;RL+H0j2RI~!q^C9ITxU1Fkj;f;CRDO$}cZ$TA6dVFO>%h%dyWk7E(t~N6+REB6(gw~&8cqPr_73}o?04C>{cPA*X`l7Swj=kn^`DfrknK(BdH4n$>C`0P(=_=prESybt+bFi~t&|)VE3Ue75;FF5!m)bo__HdkuIaGSQv+$@i7(*x95J%Kyq64Wctx7<xr=Lq1?M)1GR%?t^>mawt+k?&z_y=$4%*Ej%U}p*WjfBI4}us?brk<~auf8nvDIjv^ztpVKm(13Q!*)iXV~ZxT^^av(s!>DC4F`KU2lg$y}oALYAjjmxe;uv%8WBq>04aN`XyhoCFuWUI^h3;ck-R}{mrQ5Vd3alv!3>1>rD2hdQ;EZGYd{5Uut*Tlg7R1tiol(dD-sFZ!J&wb7kk&*1huRq1#=w7RNzDg#N;Qaedvsz1E}qS^)bSwx*W*yks|bt&}^&==wuegT?I}jj;s;@Ai8MGa3`to*-aq_lnOVHgDizJm3BB;lZ2SYKLeqcRo9T@TMNbgTL(yF|l~!YxIp#^x}C*OtHZRfMwaeQPu=TxMEB@&z^KsHe;iQ^y)%?i=PJjkz2lIV~N^q&`yXgHoNLWi!owVqpp{-qjvn!h`ER{;vb&6+mw)<8*N3Yvg*7gzAB@nCr;GB-q%o>nV#0{Cl}2@&3Suh8p2O>GO4@t>dJ=wWAq~Ts-q>(5p(?tXDI}};s5t+_?`@XL&P{&ufhl4^`B-p<##%E^f0`F%KgO(ossXZ`CPB$Cf<{cbDxLK`DM1f^cHW#+Rl*FStQ%hY}l&Ph&bf*;y#t!<}ja(h2duUIK(8DFOEBjEIQecs!5C2b>katOO&L8Man7H0td}M!`A<(zxnOHREM3OF=Pw4Z4%|;RUD#2W<Pu|??!L4mAQswe=Yo7m6Ai-58mGB_-#f~ua!+;2084Cc_Fu2`@iI35qFQ;de<(xv&WvK&@vbZi*Y*~6Sj1!z>8XQs;v7281$Vei==J<aerxbO6qdng{ntqFi22~JqE(ER-tBo;H~MonS6JR5CtDKda$*XztS3<qc?OUMY~<4NYgCHR@cM^ja(>kaOQ7Sw?@ouCdw-lkcsegG~m+n-vJ1&3NI>i)TpH~(C|Y5c=bL?R|7YEcHUPqs_uz$TIUL{`#1-$7NElhh(*>YywU39{H_j=-pi~ASeH&N_rvdUL13U(9f)rp48;x?Y2;(A0gWPmfOflI8u!atyD#F-#fdxnPJ=ma%)?zx8Rn)E^M_m@A546#mcv<71C~Ke9Td?)o}ZeRT7JQkVI4rR*ScDZ&PycyuHI>#cWcmNcCO>sC?p#klxvLOb1J>XGvAPz@{<rfY+Y{#U6Iq`R?P}tdo1mSTda3fP%gSnTPe4mjb`=pC=Pv$3jP9erQR1ajqaSrm)HMAXgGLSJ#Rd<Gm=_WKKbpjLxSzUk84@quy!y!#aN)hDeyfY4(`o_U^kx<r(p#(bYq|LBR-l<Ygly;J9pg7XveIE107C(=ke2WJL)*EVa+4?M<I|TymNeu$aTOumRHbwM5}NC%{8%Dp0qnM8V<LJu&#WVNY1{^W6&b4+TM<QGgu+xMug3V^+ozKMyxp-hj9A`jUS6!venjWr<k~H3+0YzI@$Z-4cI@>4&XgrLyZ2bwq8}e+{jJ<-}4nUK3$7KsQlz^OXj+RZ#N>h!0*f;&Y^dD?WTuUF9rb`sWh6qxi_@5sa3oEWsVisN9P8N!rQIq@3be9O6xU{Fz&|@k`XJvLWP+V6v|U8rhfNlq65&!cCOSZq|#v-P<xOE0-PZrmZqmnPTR7oe3$ZjoXX32YrUOb#*cmr+x^--IT+F5JbGQ)4J}PlyPa><VJvt1jjTi&7g4`0G9j|{Bon%ewf8wUpy+M?(crq-U0-kctkDYa-`;prU9gOAcGw!uE<%_Gy`6!F94BfCUf!_AbGJuM?(Q`NORxQxjeF*9q{G8_d#c1$s@tL3S}W2zE$w}C(mzL;ay8YyVl9E#*@exTE>?M6w!h|foF$wsHOrb`$9@ob5P{i({cT2Tt}-n`ef6DSgUz2=vsLaAeNX%LpVRdur$-6+Jh@bTj_XqIS1R>eoY>9p4XGK<(0q8=FS}lM`8XI3zlBg1hbDXI$`f8R{MPolj1m1G5xB>}L2J$%ZFNx5lU*;npT`fj!0RE)*_~9s)qZbSOAbQ_Zs8YdPJX?eI7|mGb=GFSH4)w>EKQUaIaL&cB%inEX}5OEzE!<;w_;rdTtjY~I{ra^eq`j=pL?HA7It-6ox)6sb|Gd9D}S+kdKkeKM)T$(OSI3tt+MqQh4rY;;SsJTxk^t;_UHB4+2D4!73aQ-xhca=IPvzT%G=0ZH%6f_L7f0t-Qng%4k-ZwC9hehqwD=fZ{sH>>J301RP)Vv&PQjYSUHbA%!cXdcx`VJxYK*DTITq^uzy2ct>P<;AT>f5vWe!Y=ESWpjzj%AaW>w&|K74gqNbK7^}LtI1Fam*31lGNfo-ggZi~y5wEKH~y<EQ!i?krgQQ*$3C0Uz7gCp1gq7$++cPbvmHgYY&Ep9X6(frl7)#CDlr3@I|CqMkdmD|x^#(h=+7T5{_c1?KY)1C{K&);G5JGC$Vm^owYF3qp7pu~GOoJ+3Of7V3mbQ~_5k$Pjl*OnhP+fSN^?o-Bir^0Gll8wr1(P=&4>oE#V{Q*88^*YR_$P~RCC8JL^uY)@WE+O|cIyKCD=jMV*De2Z|QR=A9vFGzsURLwEUzwqe7K|jkKJFFo_2+Vd^QAXT*X();FUbL#kF%rsjyeVdC{+cwcN=Kej6Gofh1X5X?$0X2<#NWq`<2nAhi{5tn3nR~S~FrpdMtknWN`yhaNiu5&_mh|=C?C|zGmHZ3E%MxFaLi&?+I-m77U!WBD(Fd38a65-^&Fn^EEM(yR|=byF|6m)`F)*G=3U!3sR;R5$r7K>waZaw{veS93lOuUIY-I*ij=io|LC}ui(Tun>oZyCi4c=sa>~Y18fg;JN=$4=l_}{wezRB4O`gpS(VolP)3ksK8LdYioA}wR&9FZ`o(18+?9Ouz?ce~eT#y&)_1Nt0)cP;ex_(I=ymrPNk7SKu<s$htsAhI0Qc&vm9!4TYTUD1%@r+OX_hi;Z(z!hr>&wZ>lu?dHScv-%Q-Xbkr}jzdoKt<WX`!_e)8kJzv|t7%`w;ICdBeP!V<^M+ZFKpL7&91FE3}mXFXa-D_!c5E%o^&$@`zH<uq{dNicJ^qIjMG)6~Dw<mxgxAZs!4X88h+g<!q=ewCH|UhZ&uEcf`q^IEF+!RtNNtDmr0ZXag7PEJ;3UYnLy?+$3-i`O7BTvK3m{8W$ddt0<8;D$2%6()3=7meq0YIv~@PHPOj|M2k7Hi(-_Q)DsF+>xVncI9tDfAaAm|508gqUlN0uf1>6XtPZgl(A8E4U(ZEuF8<&=g>TF?MQpq=IBej%fR^)9zUOJ;_oW#Jo|CZ`k^EcpW|LU5wF}^89B#ySGbw{3Yg!Yk^fNeVa4)HXSk<n7dKbX80^9gnBKg$*z5cGoM!_OhjOaar(e2!v<}W&=sHWIUXY8O_iq2(#~$$ZyZ~KRB>h`2x2@Ykra+1L>XlNNSJ~!_AFU7LWhJrptts|f-KNj2n70G_>_fvVx;E74W7X<jI>0;DWVGU|b`;9i$~bj~aSV?zO-!JVSCw~V=ht{|soqe3E~{96vz*U!fVe!?94*P~Z2<QNXYZ6fCj3WfG)DWk|Cv8#MTPghpc3n3M6S~v@wO;=n|udhkW*4pf83t^v3Uy1Rr)2gwe}OMvv-XW+I5<)H-?$hpZ=P|)f4k(NCY8*_u5E35j~8oFY)Qs?$K`{?Yr$ix7`_JRVC|vs{<Y!nHo<jv4NOkGexGlZ>9o3T$D%|DbvMrn{{v34DG|_uDi^^1*w0ITNe(5O1C$%ozeU(m~dyNSk6f4!f<DI2k+!^iA|+%kdPGx+px*U;hj2}?bIrNQbuE7CNl5VNBb1`x5oH8ZtCop&oqAszwu{d-CS`u98^ASA>m>mew?ezh2`;r)27BohRy|@_VhW8ET8ph`_7Mh)l7i7t1t%!*IeHe7Qod?t)^4rz8`%g6^^Oi)#>^80==C;ZXV1JUkt)8P&<1uHD|)OKJ-KA!LleO{+ougb9ZxA!L?rnbceuEw*QMbo1YvX&#D7kU%J{wd&0Y&eJQ{xkr}&gQ3WmKx3maE-slT47J<E2If&@TZec~l^>>yx`Zq#*jcG95=^Z~oFI-&<<jNK4yc)yt`%pl+6I2RTiqMfsY|qYWGoXf6=veK3|H~LwWplLjPDofyXg>A-nwcJ=F(BW!!##OMpQv8PxYw(;hl*$%2fYFTcQZ(O;X`OoOu;6;lJ3WRQyrcr*T!#V^1k{8NOH3nS6ld0-A@ttF`sq_2~(i+wEFrJ0t<c`vzIctz*cW6_aox&GIef5y3$4M;7qqpFKQR5wOA}4N<>%VvX3{Xy;IvAlVBZL47}e#xEtNXmLE?W>1OZYOlgLS3^um6!LIny`&#WanhqDtfhU#|+l(kj%usg}lNM#Dgnql)Gh~Jc@g}Rw={O^*-083LyE!vwo^!Q+Rcq6?^`x2kCVN(!^rkY*vGVsIt`DI+;*BG4A>>!b8>|uSf}T9qnT3thSd;oFai|DOWw!^54t;VVWfb*dm+ZEOhv_BQaQ(PG(=Pm@{%*hCu74QqEUa=fRS=l=jrmG~3{u44WUyFG{b%x$WzgAg+RTw}z7P=MfAiz9vY;N{3b1cQJ=HO8Qmy}VP{C@?DXyTeMNKB>pUZ1q@Z8|psMJlr?><UaeO@J)2Kzy$Hdp&#v61?f=z@+jt^Z&ue+d|Pe+R_kYEUn?LidnuC;QO24U~%EKJD?CZ*ATiso{bKCDejJPUN02=)=Z=kMa+y4X;(9YLGItXv-wi$yf2&$~81=NSmuu=7&V@DDT?0zX<>IG2=J@02N0tbu{aCZ$TGv6g=o(kK;NzJZSmR4cmWHrH7sHYbXXaXNwy*=VngNp)3f05HroAlz5y^$oPIALxk2pM+(*D*0msv;WvMZpX{gjDchFf&1&OH3mR^S?f$U4v0b!ux?|mfLi#wOcV0Aa+*g}z<34Zp#&6eH+!+_gCvN2S9k<>03xR&=Z@%_ikkk7i@4L6V1+LDmBshC6QY8{Jfqu76sS@!XY|j~9g*LYTY3r~ZHmPs`x?`pJkgEdaE*K6=Z<mH{e{=8^4vwFZdh?i2S#Ltmn6<FGd!=XO2wA+iO&R^)yz0Gd>qvMe<^Wo+EV;w=dZu@9@Mua1>EGv-XREVP9-STC5H)vr!O&^OZM<=LZ%?#}LOSD{LHBVH{19{O`V+NJ0O$U<ebCFHv6jV|F@Jl{ph_}hUk_%73=tnqD!v&)DNW+f@IDx01qSxoez%t81}R`IX&pCj@}jYE^M)m6SiSx08SP>Rx>aH>R9lommv7M5`%Ry*v|CFBbe{)&!x*-47d#c&asMJ8?1%cc7vpg?iGbkf=Xdo$?!thfj*NVhmI*n?M=pKYTY52nAB<L~ytY&IQh02d@>6;+Z)vG9j@!NYhSh6Q37~Ilu|nng9UIJaESqmH<LFHNcFm+^eOcu0Lyctaz{v>6&*#T^(ZIP!y)vJWx9bJ7H>WwIl7B2ZhwEK=UX@19*pvHti~|YFf2l!(>zs3JBpi)rG*s@Te<99YeXMUM{-8_Sos+SCNS7XLH~Fd14uj(%qB)4%_6$J@c_XE|RjHe5^q0aoPMeXdLuJO!5`XM@ntTh?FN#!#zjBc+v3AW6kj64pVr1Rhq&<@YkyIZ(w2z9cz{UBB^v*omG&8AKi|GUb3ct~(iR>Cw*XVNeL3*oqL;d#Q$ph?-`ikBm_?Ma_A`LEC&oZIu!b?X5^N`JO&&-DfG^!?4eGluLpOuYqC~t)Z_=m6!_}xMG%Q5F<qVDYI6I!<iuiA;Ue%tUg%vVxU?K<(!3kHrj>d(Upy!yX%hMF;2e)u79rj8g==nz=4?>bX;YFvKjd$<{c&+M*~uBgGP(<Zv&d%~&9+SjHc27{%aV5>*Pj3G%dm+<TVPU51y<HYaj+1udmHJvwwKIGL~-J>~fZJNEy2M%9(JvGjJBi%iq4(6}pcBFb)-b!DG{SFV_TD;1d;&gxbLhf*D%fb63_)l&jTlXf|R@f%m&aoo=++WtY!jmdz$+Z}nU(3E&uB>=+_}W};nWEG1`H5Fxty`O$%TGwO%c^OT_r~;<2%nns6@UoRh5h<);9s87bWIEIfQZUK{{QXFwU5<Y@ATR{DxILE%I)!YE$``x0(V=#=mXwu7w%M_C)>yl@AA921_ay9Y~`3AClp;DSE;zVVrlTzy&H@!eU>shTXBQPBYIA&@gV~F*m|kV3S}PG*}dhme^@x)@nQ`lWwE_c{&+fY=7Eq@N8jP0GhpX9+M0d$6P8S4cVvGn{4snx&YL=ZWSZI)y$l`Vxn<&b<ng2gc1b7Qin}sxOk<{n&I|f{pN?f|oBqzHDPTuO9DsiI^|fQW>3Xz~_bdH)(u|q6)1SfS<S)pGh#MY${ZsU{<a*=g><w->wS8ypSfu9Qjs@FZ)}8$(`wNkgzt!2Dv(MgH=JR;DmqwRF9@n>*zFmCnv*<`;9^#+W^U$jh97~vEvAOQ%r!mk3UbO=m-{Q2|6V>{YnHaTnyZ)r{YzhOp^IhiG%66=@szYRT7iETzqIOvS5##=W-E{A(v_wsPnYdM~Hl*jn6&+@+Uzy1)u)E34FRt9JTZ|RcL*eCd-gW6JCBKI|>1&U<&z?rf`-R6P_*ftGza`xPX;&nZOv_0tmv`MMpFRSlkRJU*hhFCX^hi`DECqezJ1Abjpe*zHe#_E*zn5FIu&fi3-}o*r^QFq(-!f+`<1u>gmDD~TsiRBv0hx`>u{yJ#`k*Fs&J)=^y^~AzcNuT+^IE|VMVYn2TJ(B}0X2HR#_;X4za7SH5ks7*?ycH`>3Utnd#-<M{^%9`ZAT2T3D?>D+{VjGTdq@HcB|={2==X7_tlwRNBdT?D|Qpb)V^-}u;o`Y*a=ii-c5}uwLBUcMva%#D4gqvCZ5*Vyb1>?ztZw|CymZf%|f0)Aw;{K8mo1IgYCM}qW*yFbj|-tq{<)YOAV~;5t0es$}nWnmw9s-Y`Lg4dCp$ZyOTt`4(ZRq<!0>z=4tkvF0?Ozb{$n)5EJ!xVSd)`j2o&`BaQZSw{e~Lf#MxG?`k&E%6a@k*Qt<uTh=?J);4Oq&h>f3&!p!-!_)kWa~rmoZ&>`gUGwmdimwf;Ha}d{s^2IIZKK2@Q{2VsT*Vl^xhxm0tY5#e)}AnW-eWyM2AC`EZ%y#f6<lW73}{Pgm4E)c?hs!PuQ!{|2ZQrWtAP(z*%-Q?$4Fu?HdTGm!zR0YjzztqQX&CVq5dG6B`5%*n(*M@JwQ?z-1}aG|Cs`09${Y<fL%Tc%m@am{oQT$1L8|9{D6!35n(B1_S=l_P|S$kkJnNjxvMi_zx7y4>2ywt)5`92OQYlcR-Cr_hrPb%MzK<aV^MK0VuQs8he!hkPqb^v(^|9ty}5d2jWLsg=Dzo_B%7PDHA0Bv+3<(;M^7)(*COv9?d9Qgtq&T0n~0qA9S!+ENR+X_+5Bn0cic@F$y!3w@#fvf3?FU@m&%Io9H~al0Me8MP#6<SK~8e-vQm;)>i~}OC+H3Ei8K;hJtv;-ZbfHLTKvD>zO1`#BU$+Uo^$^L9WMkViXo}R+9;crsKr_>QeLDZ!ypM!u(%l@Dao>YoJkxz6K^w<b?hvjB$G_;*iPooB#xK(;r^5>Qu5RMg|Dl-0W^RHNhi6V6PqN^UEN(>U0qdORjmu5@NsQ)<#6x7Sbl2uLhrgZGI@4AKD(HbI~FD;Mz&6^Ppup}Sgo|53$G8KT&xOf6Qga(k<?UN=;>01j;2RvW>0OM9_;N57thTdEi2W0cJg9RZvIR*6WTnQJ$rP1tYb)CKQuB@+_%5GxIJ?EOy}a_k^S0CayxpkT*wZ~ou`K94)rW#Pp_Zv=vhB=ZhAC&{NmKed37k9tQ=UE+a?cgNoRy33zrv9M0?lO#SL+0D>86;sAIpdU0L1iIUSiApFGeb#S1f~%$fO9tBdVtvtwsRCN{UmR?i+fFuK3z@Zs|XWql^Sd30P@939PUoJn3tmsZq~#le}jwPTkTJ5mdMlSA_1)pT@gM!I-DjMz$8UkOijCll+vbGhk5-SG=UrGcZ}Lu*sx)zyjAKt6PNqHjJHOLdf|5|NeC_=29Ow#`hQJD{GN=<bVj^vRLo$<u>}2h?N9%+|h(qmg3+=O&_~YrXr|!`aBbP&K+W+B3dA7++b*EarQ}vFdSgeLNy`iRrQ4_2c6M<82*hE?;gdj|dAxmqtfMrnk;6T#z@~kEc$KqD6BgbbkA6=442YUhJM)y)?2~K0MnQ?oAyzGSYkOaO~*0mB{MA_M&k9$cd<4Ds@MP3reD`EpcKpC!apBazs2cHx)W^=7=~|m>-j)l5(Oir*|F)KbUyFoa~LAR1S>g&c&xrq)y3FACii)Bl3WFN!}V%&&IO@?Op5p%3_B&GktPlT3wLl*G?R2zYxAKpS#$3O32Q}rOn~cIATJR?Xe@>6JtmA@1GvtJh3vra40n&TkM+OEOZ_g(T>}@(pj#IN&{n)U1BvdTNoQzUFe$J7wPFJZ1zU;$0tv3MYBVSbn<dvD7=4df9K5h%;1Tx^@a6Qs~f3U{9vzo^1OQL(%gYT#AD{F<v}UbcO-i$GdUo192{RO^mMPuz2)|avCeZH`R<FyQt5+*!u0TvG+&%nI|gHOlbdIzS1w$Xdgiy!C(d2S<qjtfWVPPx#kTH9qNA^K|KQO2h55Bma((}$<N1A~=j2P1)5G(N^1#K}%~7Rm-{P5pV?!6lw^QN8@e{qtuC9YqCvu_f{Mw~U<85b$quXtV4p)0SXAUQ0$2Tr@WcEwtY}ef6*15yc<ksc%5v6-}<npmpc<p$6;8^d$%0XeRvUPd$+|km?N@Acg5Z8}(u3s9Cu1-cv)&0@p#J+<A7YEDbxsm+H*`l=X*tz!c=xTLpv^0IRXW{tl!Fhe5r>i37<<#k@zP)xXGk)p(#r)RR>5fyEkEwlPZe&rtd}`m(Y;p5wb>^%n#urKlr-zQMpcQlGRQJZIY;|(7yE{I4c67Mw)X|=vig;8-tUZ%1B*r?<SNo2aE+(p(^6)}PT--d+9YtL6(B!moLX4J+`wmS+&K^q0<-_H3Lg<9JSjqKG9qv1F`0&1?E3x#s_}n>t_C!1vIWQI7H+-fup&Z<p7`Zqx(;k;1a&=$(exWd(UR+q8pDL{M_I6*06?@N>4^=hkVqt&#g_C_97jx4?srZq})am_WTcZcXw(%o#2c(73$?dAxu2*6k31Mw$?Lv8Ba4I(@7S0|xqxT&;e`+)zIdFV`b$<WMzVJ+Y>Bw|jD7k(5@ak6P^oewL=l(<I53c39)<b8?qnFRktX)34-Ff!h=A7EqdH8(fP&9x3{OU+`bai(4VCC54sqxKfOx!rXAYV8Tof$y#WFaXGURpn|6y)$$xou{AEj=JA11Bz|qHT*OR}&Kl#Ps^U!-EGW;)0sIIFp(U&90ZmR*o%Xi{mG{g+rHS&P@)tT^=}JJhd@hQ8(wZYX=u5qMfO8i>bBo$(fGC=D=#|l5qA&erCEjRy==dQB}th`FzJ@y1SxYK6ZKE*um|Q{Uhq=?AXwSw(+TTsdt~yePCtleCAkqa8f%r-ZoK5SB7GViTv=f{U@?$VRyGxw+?Kb+Bg&L)~hEEi$mu!E8BUv`)y<2Y_UU(p<TZFRBCQ?YcbUx>N|d{I5aL?s-7K~+BdIN+sBTmr!O8kcDnn3FnIjLsnOW}wvJS$w@b>N$(3h2_8lKO*0H*hzBqVj>h!F1P8#SvJh&d;-pHPa#m)#v!~0gICX~aA!yTtDoQw}{4z9(IPxove%XduAYv<1NjH?4fGr8>PMSb(={?3zAs~dw|tCR9R_2QtkeR}=)@b-G?)WYe3)5`ei)!}3&w|?;$cypGE=f~Ez4xSx7FYZg67!Dzpe&$g7P^P+n>d1*UZ6L8eIDb|<r3|cU6TOwFyl<?pa`;k7QpP$@WjALNr#fa2&9xQ6$4BE6Cx^28R*!TJ9G9iZh0W5$`gmvJVr(|L8ap>Qx4M|f30w2=?bE5m&~&7GZE?DikA#ZlV~T#Rw5adX=ELcOk@WeIq29sqRO-Oi?8RzlUokm$p?aqM(z-f(czbeHS{R5<guBjNn$JatVn?IeiLtqbsYB}r5ALsY6?!|;6O~ere(1E4J91)NE++=I4<A|S6X)78Cx_PdZ63;2X2ORQv2?66bt)4+HneYJd@VAox5=Y(#nUSn4<%A?DPS~_I+1TXJTjb(^<7k_!;)B(wxfqa)t)npqr<}4#mP=tlOpAAMVE^m-K%pmRpIc#?vwjBPhXmx&kNP@#d!5{+vNFdb^3Vsx$gb*Bl`wVrsk$I=lV*q_C&ROrgvziExVl>2`QIy*}2o3l~aT1sYBzZ#-hDyTnf)DoH`}+YAcyaAvrgeIC**~bvkxZ>gpUG9GPEBO|}nR){d7C%Sf!KZVsN?KY3*9K;=X>F{I@x3loc*ix<N4(KDAj52iu~M|(~mX|Jqpsbl9Z=tt%&C-+6>#*ZIs+ZtK#(GR9_XA<jY*H3R0a?y@yVY`qUNQDla%q?`3Ru=akxYQNbg!TEP-jz9`hEhioxz*Lsx{B7192prK=s7(bJ9T!XvMry>U#P4_j-5ZP&L2M(Z$C48sk6Ore|ul&`GfMl$ZYO>xp!9EXiF?kuFh?Ztq&gSEDoGMd+==ANOkO}90^C1zU=Y*VDxA*)t>4Nj|$mBcu}35P978c+6OAp;^O{TRP35M(lxARr>ojr?{;zM$f>c}#OC3{iDb2UF|iq*T|0X8>`*Rws!z|)_H2!Gtn5o{59C58W)5wY1~>K%?F&cODhFa4$^92M7Y2I<j%0hb_ODMLk}8?-rBh>Fv*McCbum9PQ_jid3*(Vg?EK<p`ka<nIKThGnac;xj%|jfj%z2UCd2dD1Kp>?y~i$eTs|T7E)E?$c>4Iz$YC*AKDLN-zj*0zUu0Ft9y<^pEvVvfdgH=YC@hz!&(8E7i=N1ryVSnp8&bzH`RJL^gA?6Da})ALvR#`Ru7)RPW)|era?fH`lV&Hg!_|$*LZ2A!n^@O6`>NA1c}u!5k)2jT=Q?5|L+wKwveK=j4vh-2{rSO>a(-Xp;7ES(@ZqWOfzVo9x>!*Ld(JDXhc?ed2Rn{;?oXX7=|gSRQx^}l$%EskW068(|D{lIU8`od&L5Cty_@5+>s^Dr`mx10qO*8ps5BQl5MA6qK0I*d^69fD_l+fw9vCY0$&+m(L+3(MQuajRRNLgy6LXo9dUsE6Y3^`AUcGc^QXUzeUOVak&;O7T@z_*+L^>8r#HGY|EYjWMi~9VBGfG#gYo$Aq4)<kxlyLV-XQs0+ayXOf&ZN6~)19G}-tO+6Ok^dJ33rE;RJyObJKQTP;YfF;vnzxk!+#lN#V6&8az@e>Ij3Z#l&mX)*#DpZ;X{93d+oKoufM2$_eSmZqn#J`cmMoW{odbS-v4R$?!(%xw`;fmQG593ouA(M_p`e@k8ke1@!sz9Km6~NH_-Rm!*}++xJtg)-}!p){*C&FU*~1D@V{4nhkz33>0+U*6v}8c7jtSVm{HQjd`VMuU4TyXbOlqr-S9A@2rLqUN&!Ff%X(T>{h}xmaO#S0Rn}Jll)fq>Gzqi;F}SL1WYnypmj%(cFY4o<lnxmFXtJs+zWGX_tmc)trWG~8zw`L3`sdH$;o4_UYEQndz4)))t6$eYzyI>v8?`UL-}~oRe%c3RLrDuT3^k`1%?Et`L5f_xtf}d;q=Y4eceS9F)nY*(@}m+^(Juyk*<3Lt=X3!E1^;p820er?l3LKqaxN#ObF!{0IzprIB&Wz)K`msBS4It`r0NKBQeJ_6Kn=$eXmNC5YDzj8pHEE8&H$msLtzP?bxU1|czh%wg(IP9DH4kGghP>tbTaIxb*AIV@!63Csw4Ol6EkB|aVdT>K9d}qkIju^5HL{o%?bL{!oBgwf4ozgjwOf3=`2A_^e8zpO=_Cp(8G!0X(={Lo`|$BxTnyQ!=Z4%cQ_P*e>&lxE~Ci&!c?4eHnxB|J})KaMggMnE16_}o=7Gq;OE@Jd|aA|$L8aSq`Tw(`FL#PjC5ilz7UtDPCy0r$?@5VnWWjr`Ptc|G!vVS1C}dtUII+7RB}0lX<bn=I$%04RZRrPhLa01G`3`X4muu|LZMEnI})O=r(#I}CG>WO(3?=e*B3Ir7*AYaEM&|iC#K`l+{6e1F;grkg#0i8`9@Kv66sYXFI#+}{^<73<8LQlzWKa<@16QbKi8h!u08m${_tOWKmWA*{taLWPu|^m`a|vEPcN^bqBnPb{$%f8AM8DRu>17Q`upG2Z$7MF|Db;F+xjP8?LGRj{{8P*)_~cP1S>?61YOCk1bk~Evqs@ja1E1uSy&StFIF@;ub`J*zII=)uZ@;=6k1jFa#7nv#d?e)xSpd}306=ct7wv*E0*;Dc`B<~T9a4G<ROC53Fz(Va#U8TX#JaIc<@jUIRGs%#g!FZDI1k(pB=3fGw?pArWJt{4;a<jeO=;!`2!)uQ4=xSVns_UQbx@KMTJac0p>gkdpG8SX=~i|)&akq$$NwDHhX4++3hvEz`OM(JzX;@Gt=bmF|U-PvAnFW{eNrBA|N?kR;w@_1c<9#dd+-?dJ%R_XSt#kB<Q=@^@!0tR1WxTKh1$SR~w-nep%ol#(K`n1$71XV0gtYX3`U125pVPT9i}_`0!$kTI(|4qtqW|mqNe??M0bRqwYkFesOjd(x63zQ5RVPNZT5Xav1?1w0#{%IcwXaA_xl_n$f7g_Rh`P&EM}ly}I-B-JPF4sy+P6>uC*%3TebsYj<wJyT?yzw?50LdPy#)S9gE+)9&wos9$;G-_LH4S}(u&w0`di0#?-gq9y0S-XfLJd`p^=Md;HuSwU~-=~uhA9@YNxCX(7F(O=ktprIR2cAvhr_xxe~=3nX`{8az%_n{di)!le!_vxSNpWds#_tVS!NGiI`^6xTT5lhSDcPdv*lLrEtNHj~j=vygjJ_Rvwv;)V(uvNxGe)4n;KSkSh7otJ(NQXz()=t5~8w@(&6O=+#)ry5Ymbjn=)PE6bDTPX2(d4oskY2E)SJX0!hPuTHFiW_Y@1U={e;E<7?Nz_|ZrK6?y+8ugD&oVaISjJjKFJszLAg{?3K;>@xtfWt__vroZ%fkT)*AYEi-1Dk^eq^}w%>zdtUnBxs2vSaSV%cVDM!N*5RjR-z9<NHVZTqwA#oSB{HE=Yo<+$A!EhA*36M#MlBsA01yg!47e%t0ANoYV1(E2gmGyPCyehyOk@U-tq$oqchXtsWeL1BdV1$Y=1SBK*<U$5N-6}8n)r@}`&qx_21DD%23KB4s-QZV@5=0q*URFv{Rw)3*Ajw}(D+O6oi_Ai)-Tt<A@0*?H_x7$_-Mjx4T4}HEy?Af$`nP-Ey}A3|H)#3RZ~w7&^{Y))$z{-Dsa^eH=lRF=CwBn!-lGS#+h2i(^!snv1yfuvU_vY?8)YGl2av`C*eoelCxVic*NTz^XiK9X1;iK8FPO;$En{dRlpv!7auC|4FN&E;4t5P$FC(U=A(f(x8eP}q5>ml3h%yAEN0gLwIha+-1S<l8W~y2%l|mYH7X*cX3#N*h&85(?{dKTF<S;)LYo!uW&p-*X5bY=BS5+(@Evb0iYQR^puUiuyv?9XbRUPX(1=O`d0FNZTQPM;^=>b`=1%p*NS5ZjuVWcOC9wma%9w`<pBfTh9DJvY%Z0Vqk3gUT?K+7$ozmifd7m;=*qisnoZ~B)_Efq)=hKzO5a5K#l(-|h6@7}CE`77e~Ks|5YtiAVW=kXOpU-ftYxcl^z-5dAn|9Thc;I%u~>$kqD-Tq?d`OV$u5B-iQ!#E70A)}>)dx2)n!k~bM6osE>fP9(X!$pjyvH+|?6s<*B4z8#hQeM?{P&p9`kg-}T5Xzv%fD+Jlg{4Q#0O_iiB&zXB5>JIDFt7&AEKgt(66%G7mzYg|Hro+xi`{&@WTVYxYyD$wT+XQ%6+)~UnFYbM@YjplnxgGIeo%Y#46X9ogTL<l^m*;cpV011_RrdbJ9{tg?tb$w%+vL|wWm*p7e-?BZ|>JVeV1MGWZ*uh{uRLLR=iLpUq%%fjX=@Kuel<U(>Ce%xmoh%q>`Q~YI)pw@?>~bNw1ZPh!9PpbE|oXHR!O!OKL`Ni(}+B*U}&uB&8MeKEx!`t3EYfDk9cgpw;M4{4Q811P!F4;mSyzWU3>;shonF$*BcJ?+l`;&!g!n!AMmk!&6u>kY<Ll97K@N3Xlv)Ik0|#APq>vh^1tUnwpk#Bh33&;GrtHlnjy#g>?)S8Kw_g0e~?g<>XC8v%xTCIlh5~z@pX)Fx2gw38k3QC0UosN<nX_woxxv%p#c<rbSOl<cA@&TgpG8Mj^YI528<&Qi#-OZkF{w|E~7Nz1@4C)gFF|w!7N(w_pD8Ga@a%{L3e`FF&Z?`Ub6x`iDQD=hRO4C0Zgs)V}(l_WSSIwLy0<IU|>4#4)zEjXrr82ee|5NPVklVlNiao&b16tHF>}Y+{in7uKR7(IKQ6iU<DGvAM%EFa#JIh|S($srqaKL@PAKBv3F=lVERSOBn^=9S7ohSii8tddNqx2XW*%J0!HkX_+PeLTp+}B<ClFlhX9;NF2#YwhpyaadavkPvV;XaKztGr1XdUoKfYJtegg`O9C@=pe&Ec<6&8o*Ubqc@{VB|A#j-j^np>115Cy|Niw!2AaML$4<f#hMr=?KKp*E$H(G9rR+YT~94l@J9fcCwXAyB4p8`W!SUVN*yOmr~E_X(FtyqGn#E4v8qAH7)#?VX~pSw&kmyrz362^oUArfuuz|nqdn6izc64^C$v=?Lfzz!Ob2?x5W`wGP}F5mA~8f+~Fz1l-voW|=Ti3Mfch*@T}SX^_E3lA@d>q!|jPT;0dhVWr?6j<V)njItN=Y$`t6#g)e!Vw!+prdlOP}I?Oh4)K4_i`6;+C)rfR9iyhm$8tHg(q`_up#Nu5aMquD+)FTQN&2;)uM_h5xr}sj~RbT2t8<j-IUTrB>U)5Bp903H6$UX%N03iLgy4|wH;#GDpnN@QT{q~zNE-&Xa@3#qe=L!Za&p3rBZIwd@44;5asyH@<hr8R)Lm~guArt+Cpq}LN^kpzGwW|@2h|KIg$11x4uSu78tLw)qCp1%WrS&KKu}EYg7Te^_g*81P5mC|Fiz^UsES)*Phqk{|@bOu9X36fMNJ(b+U^N)|xRhXmNFfMc<$=#91#?(si#h!GSa^7RYD@Qs;wMM+sJyRW+ScFpVjOU~rN=Ou#ccfHayX746K}z*?@9kerTIr%)x9nkwjS!?Br>iIG?`o>(H(vFzx?g^E%Ep1D=+$NG@D&=FabRp@|jeU)-ZPFt!j2TMq7LJSvvGL2dN*>-4YRkWr-{DDmZ!BH}TpjdV%UYi&pNRlc`=0m~;B7UGDu{n{$mKY?ML@a{EVsK()Do#{KH026oQB*TJ+6&f{&1g=}r!uln>GuIjm7tmt9c^j~!X0A$K-kvE37SnxS#HKjmi%@eg0}SbC~G5Tbk(IQ57+P(&mx)_-;gh~ETVtIcz7n>z1X^=D_+ct+bJ&~4)*0=cA|s>3x-w7$puUfje4yqa%Pi~t6s?qzhs7+lg_JBCc=Cx%YaxQwJbhIvdG7rKl2fuM<TrC<qg5ZP;EA2qgaF(!5|03rZB?DSCF8qqR3e~W)Dd3mVJX!E^EUUWCU@I3&JHMU`bh0@}=^ofM+*Ahc9}-ws#_m2ah%&`{NF*3Sf9aFpmWd%Z}*56nI3++eHzjS~x|LVn*2zXn(!vpOvW?>80PB#N0cj%^VG;j;M)9_!TgZGNQnV$hZ+VYO>86t1W60RRCLbD(G93j4I$;p&B!N0A?_=xabm==;kkhbO1*@bXznJfL4l#gGI5_=@g&>zKo$CSh6c4g}c}th6REtfWr$}53b8<87x~FC9UdUNZ_EE)I&|tEz0~{;?ze{lY>8FDFt>l0Rsvm6lT?eR4VG&MMU(qvu`s9QEGwc9##>v_jvs-g@PcOb=ljo&b*P{A?vcH$(w?S$P5^pql`uZqU@@S<Z2{!AmS-)3Zgg08i?|=g{6)RIm+udzQ6%BEVyFExyKO|2Tfo(lSlu+X)aQj%~PcS^ViMNt7?WXgmC(XSQyM3$S;Vx2f$uJ)hxY%EP`q+bC^*fUCg7YhAv_r5`@zigiE0mOjk4w<X#wx20~saC=%vf!W`d8IK;TD3r!>r-~y=chQwGx;vyv;tH){vETn!B>ny0}jUfWjv*7SXM*v`*M=5AUw3_{|2&IyeoRG>ztCM~XI*IHb@TnOv0jF0Hvp~oNlx|Waqxlb!|J}7ysBJ}474{g?bMM!K5*Rqi9&4iBgP+o^-$(ZXyzv)oo+CEmS2TrE7^z@T0y{yCrv|N4wMvF;iXHyZsWTHZV}8P<SP~#Ro}cj6S0)2QP^6nS*#!s|7Px>%gMJuY5*zF}u-&MIisIRj0i*#8R#~wma%)hq6Pp@Kd_FlbH#?t{hR0*`V{vI7t$lCM8kt41K)H4CGH<W|6>OZ$%jMNzT2XU?W6fhUY6=_(0sn`UFNzJo8A2w2ZRdbxWDi2zo95|~w6?yABotD@c!{2gzSprd17i*N!b(SPqYlu8<t=k4Im~Yi62zziBJ6KIsNZHdt=bsi4;eW-7es6WK7uq2Gzb7GA$}_%rY$x{kHLXjM-Zq3tso~99Kc+zA!J?o#cPdlQbXb+D8X%uXvZ@drTsmZmYU3&=)ETfK>-(FgC%_-KU-AoT=pTqA*Q!lJb*P>qch^GG;A|LHE-BaoHR@xdgws5o*@-1<8l&243~j|i`bot`de2)?B+LrBNc;p9$owKAN6-$>^*uy+&lMvxLUjRr1r&k^^a~5SI>VxyGu@_Uw-oO%cmdhJpN1l);pxIpEG_e+69%3GEm-98PQlzC}=rw7<d6s94gZaZv1pKon6}>1j^MA!L7X+hnHuEZ7Mm9VENP>OBjQ(PwsC!2&)ByGJZ|(%W}os4t2vOh=zldjXW?*q%~HR9NxzGYD&Nmo|*mG)K8oWN+{^=fngLReZbwN09nHyr|V>~qnZRlJGRl3!JGhtauE~YiL~L;UO2#6D<Z`X7rOTvM>>o};iw8Gy@1i{*TH!>=wMLhQ>*pYeG&AaF;xgAOd%XZqcm%a21K4bW2#$bF9{wS_Y7}Xq*K<`NPkmFqLnVwrK1|O33yqNE4gyC2Q?!a$jU3SMdoDf!d8nP5Q+-7+7v)CH$6<3maaf)vn4>37(8iy32dUY`%TM+jk3-OvT6IU{pwEL^wn(E?#X5hn$%;<5M8`XnU<x}-8-?xS?9KDN|e7v3z&hhiXBu8n-uL*xlPh;39tfO5`uX2RV9Nh_}0=em?#$c@wB!8l~!^%6BuXeVt_4xkBUU+)KZu4LJ9W#G(v0vyyuoo0}o^LDP|CNFi8P@2k-lKvKwRN13O3he$gMa36ZX*ip8A3F@TBUvhC!1*=<&Wp#-t<39N&7w@j}EL_-gSBXu1|P$>eu5rb%!xe7axVPe>I6t!(gF%NIl-~Z>{{cm=^d_uxjUcUdL_T;ZPB;}o#AK$58;l_744%<rD?Z;uR1mCryu1F<K;U4SKSt6$sJliHXF~DIxwq&Q*;1n}OBEoD60ob?6Y+qEgqK-DlHHr*-L2!h~F$D$VNSB#`0a6OibCEO{<PO$JR>a*NcOyWmFY^z&slm35dJJP%V_&g^owB%gThm>@a#(0=VC0AF-K<@GWAEYP+P!b6XVRbFB0fnwk00QU`8`}rf=0JZ|7N4N^iyp0*x&76VMN?%%DSv&sP1Qfh1g5B49*uJ=No*o0f(+=m_g#6$UEpG1}2lnSc2r!F<XrqdIKD%D+Hh<YsB${PAc)C(f?0ouy*Ih&f_a&%3fXrUoqMNzl-m&Au-r^3uiU7Z%d{F*0RHOm=*~51Q09(<SZMni$VsoNs)S?8>6&0OZl2%y0<&5xS1Z08q9M8TV3ZF0)++V1~3AnUDI*j!59iv3lAnqlR#RHehN;M1$<T`VhfF!6WogZ-#Xr4>UbV<uw}SM9iB?G)Aa*fC3?CV>BdZ9!}@`$IcCZ^KQ2tHqFojHYcP&cmOpWZ`s(RH&O6=-dC)q?1clv@ttC^TL&9>UY#fibDo626++y?gLC$}bwu$8UYVErH)4crO0Ld1GJHX)R2}e!Q>Bv*N{cZizw_aZPeCP2$>mUAL_(AM_{k!_V-UaK$gD+~&Zd0LY=jlzfe-JSbJbYd}ul*iXyi))BeU~qf5q4^>ITEO@83F4MJQ}3iI@&;;IItbgu{*7`DgY=r`lc!?D~cvAg>i(^5^mU~H&{-5d{M&`3z-G?3q%KRmzyQl*XB2Jva!5tT2xh*`FxA`S~YW4q2Va>JkbhB;kY+>a2X<2smrGAfXc+)M~lW$<Z>gU2X!06(1UVelPmf*5b4O28`FzFIR$_pk`2x&`>_JUQa|h03Z%6hFsYA`487zI+KPF=Y?nF6()-0C$18{%q@|so-h~LRf80lUfS0&IeH{Kji5YH^H9IhbWJHtjnK@lDot|18JBb+Geel8V`(M>>-`RQc<<670K?wwOeE9*AoOka&tbcxuD=|w9Ld>uBUgP0)ILyptZIjcz7I?U~UT20K)UG|*z4cw~%ReJVZO|go0?ddhB2w->ykCFd((jawv{EYj66IoPq6~ciX-d|884q9OGCp4;GCi7LMazTK0-8fvtKfBmzN&>4a90NyWFK^FAE-($x9R0G*NXQx><`B2wC;A?<m@4a)Z@8lIecdTsQ8-*NF<umFLEUydm(`@@>#`l#Vi%EfTkfl7d8ejS2C@*Za7jCzb9g+;TLB7Nm_PwMSC*d%NiYy=S+{GW<GvuX1T!$VsU-Rf6|Xygif|2^kUO3KBgi@t_BXZ2(oRHd6j1H@gZ)e6(KOk5iT0TH>6(gCO9%~qh?I8)-r%n+DDDIqN=wWTR8bGdZ_L0<Fr>*$_+k&<HXT-&g&(A5|Qkbnpev~lfz(CVfn&*WNr=?OLFfn);ln-nRWYlhq=%+9WZVVcT$V;5Qet?C=Wv|2MYMgQO3~W>);tfM6XWLLFFVSQisrz+t$&#O~gh30g7xs`b5I;`Jo>OxRM5cbRyEM<m9YNpkez#3{oNpb8w6X#aVd+Z?D{e*&BCBQJgS`&#%?4y*;*>r$NL|ukOD0&CZisND6rQ<~Q{(uGByK%2bl(=4N*u->TjHwEo`DJI`OBk|ZW#=kb>>Kl-8m@Lw(+iAa2`9mx`*MBnSLyK{_S142r-cGnEHAJRoa@`ErkA`9*y9~woh4d|$hP{_h|q7M7&D%uM5;5Cf90beoWMB;0&y#`?*KmKF)>FxUM2fJTB*?VyfiBPj6NaWak_+eSDyq+AHCRx)!8-4F5q*3lX{uZ(%J-)Jc?Vq?Xu}eFnB3W<KgqlY?(u@(?EI2GG(PCz)KLWxy$BF_kBmEvH2(N6Lre3(>D;SPBc5D1l*;Gt5)8&y0CNLLOr*nvra!u{8Y=(4{4ieBNt8Y;RqqW>|v}~ld!2WPV)gm$0us=M$g(%_qAHfiM;{#ZD@4Q?4@vGfue?v<SkqL3Eczy3*Z-MB0`}bI8fALBE#v8lu{B8Hn+qIkT?>xWhBqG3C!Dt)LAf@G$iwj{v6>q^vQV_(Ty&6<-Ai)TnWddelO=9FAf~sN*2$lx@YkpX3862kuc1QRt8oI41GtwX^9F<R?8_7Ya;lg^sHL+T+VHlipIKK?GtV7zFP{7BivRu8oNW^OmCwC%c4WWXB>ALG+G%f{$aQ0a2cT~nYv)w5h5ZkagXO>8LvqNTjCr{9vqTB^c@?6L|PDWxg{WuBSh+`MHLG2DIuU-9n?dtcnTc6dwes}NZKh>^&wR`t*?GEIQBj<BSC51@4_a5#2_^5v49g=>i{^CmgpHE3xdF{$)B!irx;?<siU;EQdk~N7(^p3D`R}?R#%!wG0*-=gwLN?V7PFUm&IiSnLY>3lDIMT}t$;09NI#Im9>O#F*lL~ywxv4~Q+L0cB8@X-^JGUJqLx?pe9-afsTrL^=Lgt1-muUx4QEp#_rg70dh|`mR0DD9<+EpkC;Y?j_yxDVi6>xx4p%6?!Fi8PS>;n2Vr4|%f6Ecn@Up-yWSRpvCOf3lHD=pd6<$TL)7H@b>+C+fxFu2WhMx1Z?(XH5h+^dC11ho<F^K1mYA_V8%4BD|ID~B6Iyk~5Q3rX_5HU3^lg@gt@q~L!&!b#o=#Lx9c2x(!!hRlejuO(o=3cE};ztXDOXeNI>tsto6{J?8CXfD@&eKXX9w9<(3fs6oJ!)Y~>e~}DU{G72~Qd(A%ORG`=X@U*9(RsK@7aZWg53E3sPEnQVO0KM~Dsm=fJjb+({!ii1*%ejO%h9A(Q5u77jw=A$*W)!WeLb&)Nq44PFC>;T`6et3ul;6=w;|2g8Y8sRWZA)Re@kSr8G^kqp{DbaVLEMX3o#gDv%a>59B5xHR28iuW}1@;$ym~;71KAsbJ0YAPmrNC<`v<dR)+@W!se?0{)T_u<WrFV@>?nzHF68o=^f-($WTJ^QXz#@SR!A^;ik(aTQrIyRZn+$REx{*SXH^Qq2^Rs+e8{O$TO%9{M;T?vbK(PP5KKmHATBq1VbZ^Qih`IP4_e`)RhGTU|yu+Fi+uO0%+Zu6KkR}2Ou!$1&B-iAl?}(XPGoZ9wLsgyo{4txG+i1AJ~B37J4^8_%fVBnf`-jRzlONXvwQK`BgCb)t@o4%PyM{uqkqw$w;aWx}$QMDYLOyFj@$>1y+HBAu-^PFquNd%(7@=%oRsL2TqX-Ph2A~O)IuU#CBLTC$I$yyMz8BP$a3dea5}ubh`yp@O88BD3C$6;SP#OO~Bi=5Ee;ju<Zf(+*`j1%A7NX^Pojt(j_}}FqHwPv<osVe-`pedQ~o{dfx0fVJ#;80P7-FEli>KD;VgiYT9^&zk8ekTWMxrHOD?rD&QG6Fpo5l!m!O?RaMr7Fyi@m+(Bid1cpIbCzHbA{YpNiWI&U3fJ0(1#V|G@qUAxNyKysIta32$Y67EiMoacRf$15cC1MT%i{7#rgu|u?37kYUMSca-IXPbf=O-Z?6q(}uwm>R1M{NnJEMd{^U_$jAv<WNVlkDJ3K~<pd9UqF4k=7M8yIR(x*tYEfoh4Yiz)Ao!Ylt_+-a&~FC19)pn@QJA>Xep@HPR90ELJ8&2&*`jzGGvwnLL@B{_rvcsOSwbn=58X;#oUEV5oQ>AajhqP^y6BORtkiR{erjMj}wq12jdwi2=R1QiiBO(j?}mXo6PM9E8InW;^H$c0*T{K9S^jB6Qgdxbf|E;~VWJWM)xHU^BtG2}`{Y<LVJ%T4l~BO&w3LBZs!bSul+$g-dI8g}}Bt80AE;vvPzWT!&7rhv?(rbVT~W^&E~Kx#pr1hr%x!Lm=W47EJ~W{~Zq9Gl|ri@_Jj69kv|;o<-BeC5A?gWdzgjY89a>@Jr)4jBzdD`R1?IrhMBK(h~Aw)IhqhD9JdHn2QARme6RP_hIJ`k(d#^5@@g(w!XLF)|V(sJ$bx~k{K08Frj&`2v%;k3{iHu1H_Sl)slypyi(NQ0Ml*HZg`x`3fF8M%=KK}d<G(2jhiXN3^Q#D-Dp{#Xw*j%4%8pPwn}~<tuqOdE?96S5O!v=Wv^PO-Trp((Nk~)gliSp3l-2~lHu0$D!cMV?WZ5=pWNSh`oqqXTLet)_mCi-djgTFs5xW4avDCF!g;d24iZRtwj|H^Zytpk26M$QzX#X6Q=giqAF5pwMwlxxwR3Q63vn%ZsiX$@2kPv7B490RWP*(JnwNtFe35Rd=qKgEs_@DjiZPyc_qd@s-_yQx)47(A5UO5OOwV)k0ZDo-&?`VJof{@E%FGH@I@;PCmDkYX%<8U%W7!!>kkr@v^tEIum2^t1wdOv1rE>-IM88&7jOw;o6%br!T|*dh(1&0Gb$WW6x6h>Tn)D(+latu5t$|!~!EUbJY1v$eZvg3<p*NAY;CaK*T9+;x(4)kqW5D(};KSa~AQX9wo+F@Lh96Wo;~Q~k|Hu(HoRzYOtFE}+aAfe?kW7?V^AGzP@h4IthK!2t{k+Zw$gwQvVktsXZt`HWp_^#7rAQAlxzQ=wqvV4#`9#v82t#BJ@-2w2Xven=g2-NMJ3)c~563IhU7Cz9V6R%z(7>Ds#ZJZ&=}LpeWMGKVrdq&S1J>%aI-%MgMki-91_zjXH+6TWiB<*eQ(>r)u*O5XWM=>jH)q8XqmPZ{P9ko4>R(3K0c9K+yRU+qEjT-9`)yyU#QUA;T%&a>jN;^phHl1?2kJAU0ODJKY(;H7AVXw?J&)y1DzR&wX|aW6PwL|gXp&=^@P!ztH?TN_hM@VFFhRS$72WWE@c{PwUS7La|NKV~t{&dsefVMR$A7%M@(wF8?f&^KB5T#Y{O7-)-FW%Ni`rH2i9;~gsK5S?f7p5QUj6U)N$~dW^FQo9z2|Z>bg()WSL`bXu0#)nV?S3X5Be=x9y7JJN<ecM=Vk%XV}@*_;?0mt4*x9~(kSfK(rU46Eh5zSLRJt>sVi76nrX|4+0$u`T`i)TE%G04=F*&9do)YF*w_i5K{2)(?qs&Ouafp$^O?m#<h;rBQ0P3aGjVFnBBol|4x5U3v`1FFj<J<OtIbu;mKHy6MY_i7rY}p-uq_PiLr7i;_QqS_cBX(~I3t4-yF?>Xp$lHC(2CD;oK-<xNRec8i9wA=wAy|jt3TfUw)W_!+LQN?TKV$ht1LsLS1Z5y96X92zPWep?Uz?Rha1J7e&FDnwL7<VpMH&??Y`}HySC6sztQDs?pmU!sKjzX1B~gXZH~{hz5F(!Y6;WDuM|=>1IVvo|E3g5x?~Z9RShz!I9^)(+N`KR7B+Cow~CvOIEfZmn)GA=(yL?EHY$#>WNubKf1S3G$6xL|{`TeFAK>nu_g~Z=+^N0&-@6Y!fV;-7-yzQH#NQaP+5&Mfhd8@`Ku7=rSG)BYi67)zM0C`pl9VrIlpOya48s({7IYCTQq3(iU;__t3*a8m^DF|Wpu}1Qs8R|~g0YO87f6fP3=md2Iz^JT2KT}vX9DYwELRnL?F4h%vJEH7Mmkr?z-@U@lE7``NMPcKTM;LQ^E;Y`GeO%Z;emuGp)A2sClvO(FU<hXOK!1h)-&u3fY-jJLVUH(c#(U6ai4lRuEKEk7iW)Y=F^Y+0dZMNj{1CWXIH=7p&&Vk-jOu)S*(~Ervg{D88?K%l~njGbwLUE3PyPxr{WpDT+tOt&$Jq|T)+4C-Ov8^@}qz4JbtqK*QfROzc989d`IhhkLowxAa`#iM&h-bxAz`C<c9Fl1p}c;Xr+*RwXn_v-r1pjWJYY#fH`&~3^HSt-SL}MB~4-|ARGW<0(j^C&P!12go)9O#Q+lYLIS;9-az2mUmysFLvwPL&}1g4-A#f>2iiw_&;t(fz<ir?wg9S<aq(SI%4%}PGmkX0$(oyt#lJzvaPTvi-N<(7n_zUB3O3Xb!0986xR(f`3nh9GPo@P<Y9w+PCivM%EkDz}V7gR+V30~4(12un?4;y%$pfI!AS>hAGK^C|rw4MN2n<il>%mG1j#C7t+unoX&KOhEd*(9DcMqNd!L&`Chm3FzV?Nl&9ES@$dBhyTP9_0qpUYoGZj6#?*)M?19OLHc{G?OfvXWcQG)tQIf)nU#IPaWkZfLkJm<~D?MldUvSJ46pGhH&wKFf?$Yd`t!42dA33Euty=CX75q1}NsZW_%>4&+VFLSjM`{ftcQwT^gc5!RjSbxDJGP+)NyW^TCoWyFYqOPcw`n8+VgA?&WuAe#$b7DbYB(heuWgM*Fo12}vVXWiG9I^D%MUL<I}TGzR;zuE|wm-BhZh;9w*6}b$l)PT{T_E<*4bb*^K6`<)ZMnj$pRxiC-4b#99#_uvt1jjE3cX<(_q%=U`4w!GvxpuY9ry~N*%_i;eMbCUV&}Dl<96Sgw#sk(d<2n^g8%ydk$@jwxVsDf*!3gT+_TB*P75hyr;-bqm{|nivvrm#4Damj<w{d}C@kUM3F~_dmu$z<wA+J!oVN1b_q>V<X+dJjun2n|nr*kIqQ9DH{dWb%Cui#j+MkK(t;a-g~PsoD1VC?duoXJ?bn_-Ty^Fiz0nSc*Y66knaJ0GTba6nwNr3zq9luS;~msjvWJku(9aAOYBx5xVXDTW$KOTT)FGNZ@V4d%BYJX21BUsHGu;n=ZAGuY^jn`VS^i*fOEn)ODnO^t#Q9nSBe3-rT7+5}aQRYh=GfW|@*IV*N#cWjLn;Nd!qFfFWo?lkcwcE|F8#37ZkahycL^W{ucE|g`Y+c4dUGT$lbMR`lK8It*m9VAZSo0iDwx?N?)`*K6$4k_W`_G~xCG*<Uban_DEPF6}xnkn`wUk*_t21SZdPQ@j;RM1LPh!f*@P#b9y<X6)LK3Krq770&5(ospr*SLd^wZ1Nw<PAhSAz*`^5;ryL{zgC>`1JC+Z-Z|&-0^DteHAE(qE$n%)w1hwBN2jNXHqybA{^9sCkw@3@ST;BP{=D)l+hOcTtY2sXt@>hL2^N*V8ICxFtAmXa&lUshaj9i+ybJnqEY7*3Sj{4N=WDMx~>W-p-uJE0j$HKpPq`UE!6?BhJ*q>b{LURSFfiam)S6JfuWe0380pTyppcCm0LJiL%={|3t>lX{58D1+)UlY2*j9Z01)*j{Hj3Bpg8JvYC;A=<@S~Qq>dMC1N3O6iO)IA5);qG9&#G5E(!>DtnmO`pUrm&4Yqg9T-_z(K6$$<1LMPB6T6_HHTvNJv?Q56V6>x|mfXCo^BJ?gU;=RYf>^HiLtgjP$WjSOl9FbgP{9Ee9RPiBko@*c1&k#$;?6`6wPbTY#y4POHbm}(2h0vEdm$>1lbK<~g}_DapkXN+CBbZw%z<UtKtOpW4KG}_*hIwMgw}c!;ER`2SBp5CGiib_Uy}|FnNu1tar(+$=RC*(2897;tYU-JjOohk+9k7Vo;oVX6r@N)(Vv8lHM(sJ_&932!$Q~+T?|2d_ei7rVr`NR4jv<-C$`KNr2?8MuXuB(G5=S9fe{2j;=>jVFFsrd;*=%9bWSZ{@YvcZ39!=m=WGFAcd+%*6T$uee4x&Y&+y>kMqRi%-BZOvYbuKmKhi~nB0m;iUuPJ)1YrF1-Wt*jJ|d)>JiTY;m7xQM$T0u_w>ECkD?PjYw6T@!SN1AIXm=kJ53efewGuEDoa<+pstD|=;UwJ2s2Wz2H=E|oF>-F1HPDKhLySwXKsZ`L%87Y)MUI`g`UW<95NV3~yTf_1!xn<}k8UK+g+P>mU@ZK@-JY3G&+9ag2YV=Lb%K|3+C+azRyC%J0VhIU_xaQ2EC4xpHR);p0W`Enoul*6<v?~!=kPh202C01^G;`0vw$rPRDcP<2pIka0SEoCxI^#}30$biIY{#<fuAFJM1Bs?UucY#9C>ZLA}}CcOPXJpNlr}1rMZa_5EOCF^^%$)ZWjOfA569?EHxX-+H0@9rkCXPLhbgW`Wt_#eR;F?<5#s?zx(&I8{nUF@0<F)zo(U4Zui!s+F#zRy?u4>;m^CzAJQ~y&#u(2{x|F0z#)FEScL4)*qa>fAS)`I#K;BPnxUF26eKZlxU_MPAFg8NeWbU_LTQl)m+c6s@U$arbJk4+WZ7%vbeZM4I5!nqPbq}(6k6#ywV>#oG+vtQ07j}RV7f%Nk>J`C3kDiZ+6?-0j>sxj)QvE)CI3QfT1q76Cx(;K^z29+r|v8-`I#^>_W0hT=IB&Bp2SuC;fNpSI(2|Bu2&)q*&p(=Q(+s8fU3eL+<aioffG@PaaXj8fe^yRts*8{&ljsWtsp#rW1x(x1A+zP_wf8ga$-0(#eg>~Rb=G}456Tot-(c;=_RsJAik}s7s2FhJWDI^9hD=NN&3Tvh=6TO@E`W|voj;ghfEd}z!^nFYXQ}+zg_?IPVL2i?Oy%5_QxA$;zE)fw07r4?aoa|`1asV{l>MIf4mL}4L`UBkyDRv*1rFA=gF=5``^`XKCIn(clX)fYB#R3OA5LGRMf0Hq_lZOld3t!0|!xgy1ThHB;v4*W(Naj*9vW(j}ph$1iYCv10p~}j;x{WYDL8tCU^sZ3;q015>h0?qUXVcyvf0m?N%crqYX~eTiQgEUx-rc17eEw9*Teu+ufs<WrDe+f~G*BVK?wXBl|0U=^-MgD;b&AfUovy627X-xVfAR#891y^#)9@f*)!cnI0WiGt!`Hm?k6jI>n-BpRXlA8cF*LoYJchmFG#8J|rNK+{W~lFlt96djbRlUb|Vl^N4H%)apdD5q|OA?$_VFymBX{mUo|iz5Db5%T0h{&g;d3z?tRwcwzxItl5c~WPCmd#jv-9%AT_~gfKO6GH%o(lf8;~b(Am!KUo-hltVS#Da2{f3&nMqezbE&nLU)1H23?+dM8<dcYpU&?dmr#-~49h#V5P(e+*e1Z+}a^*Y91~d+~>z$6wa}exHL>wEn?c@YdJ0>p$#%@y71mk7_qRNBqL(#!6Rh1}{Rvd#PYEMln~lZDL4aTtiYh&;jDhh?!f^g4tX#CHMguQX4@bJ4TRtNNo^&D4eDKj>xjh5f5Z_9G-$SAv{l{w9_R?C9ZHwqypl}Fe$)*jR#dF4T==z$*@8Q6tA_03ERSrlTjEzvFE|E#Uq@iN$e>FsnyXaaTmEBXzPgS;j76o{-Hcawk)m}uq{O}%cHe!d=IL+B&W~>Re*tz)ots(QII>2jC_q-fsjmyDBl-lfb_S1o3TlMdp|y^{ruMMU!Q^`_xw+_cWxq4X!q{J+Vl6#3nS`}uOd;;->5riaDY5d;{hpEDIg(gO|WlNu)LicFZjC(L}0>Fp0P(cbt3!{>jOwY)nT`QN^o_Pe|f+v9?Y+y?*hS@j`d=&@TqVJwZ^K+XuYRbp{P^YYAT`OW;sGy>jIh^f`^ReDZ7$H3{}`_NJb^WlDx5~L<iGEVEd)QnyzLGaxO?d1o|s6F@`w%Gz8}6QyCc-s^kOY&ZvH$AmIflVdY8`gXL8Xao<U0GgXwe%mic}*D59E%$O35IX@1{A*X2;<rs<qkAnE5#-Qj|2>*@i6AT*wNDq<PNYUsNzwzu537qzhiV&pYlAj0>{$;KhVP1q_NhQFEjLV;FE0)J8_8$FM|NeKw3nQ_e=WpY@?T^q7MZz<7p8c6zc3!)FZFp|M;1DZVy;KReXY@_MQxPQ2QlQ$I$1~jqqc)vp6O~j+E2fdw4SImtNPrydNJj2rtjii*p}wpjMh%d1m_xhnk+fgQgVl5F{ci1PHuD2dZo!hBF(q_QUgo*PgTFj!z`-nOXPu@6L@shhpjB+UBtebnhGO}Hk)ZDZ@|1B3<3@wy9hvrEZf@4mu(@9&CKV<Aw)9ZINr;dYAM(6mKM<Z75;9eCEJSEGGg~A3cB}5-csr|qbHDcKyKpgsp?=jLe1-HXpxPThy}WwUb(<|lka23jjulJKkX@~W-2(8%0seHUf+R|s2}mjya+}-~Q^qN$p=TgiOMaq*_`yq&IS4h?4O|9mBe)2a^KD88E`ne9Irj$Rw|L1fv&K$xJ5USi1}`I;#W8f39>cWfyt5*gX|!;LXHUlG&)7tEL?*aD(U9ndW%wqVh0rs+zVI(UJ@h2&EhEWguB^`p_Rc5(Q@Lo^Pt+(=k>e`hv399<RKRC=eN*9sS&8%s_G60%`}z=TgmVQb8^%}6!!KGi?6RDbL>s8(t+jq%74uQVT=4}>iP?quVZ@}+LRQNPj)p;mvW95`C(<L{1`z#{Q)KLxD}hs^U)*lU76jI|N%(-Rew1$Yh*@XRyy*SiN=Fw??6o1lfQ;un$)22S(DWfuOdamyev+CbYBRWcTw>*}%Yt5unxc`J`<^_p>_u$idyb6erJfWZP!sG2L@8CHOGc^|;M*(APf?`zL$e!w1J+tqv{VtPxM4`UV&++Kan@$e{6Wa%ix@867+lHv&1?=%WsiIN9D#@!FH8EnPybna@>la}7n^lP$<oaHwnb-bTEyx@3z-U?xo1lS-L8(40RxXBf|X!ORQS>|3J9SYb`?lf1tg!=tMEa*9k8_NUv>j^N3Aumr14A^wgG~=FeOdLW+q1CiKLAO!sJI;$%xK;fD`VA4;7=}9Z&*32SI^&;g_(c&d$CqS*~&KMU=VNz)}r@dAXpjDBz#zXw&p*wDBu5CvXaFX6aRW$V`bC>`GcOM^%{Po#d9%1}cuDN%Isn>Ziww?Zls<13}K<FiR!VK@*RGa03@oYHC4O)6{1cOg|*qEXdqyIZA)&1-Ya{z`h8*gg0nSIdEJmMlIe6^8xNvt1?=wXf<a6MG2((%q8I>ELk_-Nr`xDir$zZCB|crmdR$v&YLnBKCX!kCl_K<NF$8TC43z`fylP3t@=Z*8~#XL%!P$=v5e-*gb_9CSwVVH$3dn=!U1|um#rHvDW{=DxWPGHV}mev#Zhx~)1{T1k4?--@tF~?Rpo66Bo|OdVGLYmzAj@u5hzc0Ud%=`3UWVWMURqh#7F!#iNjEfT=!fGSZYwookTXV4XX`Qyl1afBC4*T_uFFPa2Te>WK=U8sg2ty2dsR5@QS&=5lR^$TGSy#cVFb*?>dtSnN^@&n^YNFNRH3WOE$sMalsMpuwRlZXsbngx{PSWC5J-KY$ZU_*|x6>)U+YfxQaIew*<Ap3=_i-N9kqTx<Ce0%Bf|+FZsb^bScbAx&#QtFbAXyS8_%!L1uOXo%B9$?5c%pLpYKZ{Xnj##P?*9ujc0xXKLVjFgodwxS92(WbW~_rk_De)p-}E*R{;d*wbkRd-@Q+5va1NrgMrOg!|U=pdz5Hys8i`B#?ePuf0ct4AYc($GzyDTi7Dz7Utv9Ogx4Z@1!$-CK<Bf%Oa<A>d(S&7w+0ZE0+}Ebpc<-)yR3P-Pg2@aI-d00e2#Gr*9SHN`Q-$oihZ7+UW|US(A(++*Se&C)OYFFY{)AIReOmrb|<h;EufRi-bZ!xVD20+G_PmR~-9f(39iBbL>41WNT;%u=<pAFa@r8+9nz@5=+%Gs0OPDwTk%y{U*tYHw0P*-=L~sRRrEjXpSHd60Kp69KiN#R1G%exKEu#*0Fc7VjT+K_Zna>Wn~u%zLGXWZ=|hyTsf;2kQ_q90ubo{KSl_62?3_0WlfX{vW?#_im)&YY<kiw^qK{yS5icaCqVGxDi<yA{?;KmpeO|m31doQ0LJXn)ni=Si5Z0k*N7qoSM&`=4IX2PwJtUoq~Rs@bGf}2AJ;#6RJ-@h?vod_JCAC&{{XjgKKpL(;eC2m`S6R~ufN;<?5me={HcEL+uGazz4zk1z3bn?P2V`ZDY;k8b^@2)OygtfND#-F;YHC-ewizy%@^$MMs(O6?2NFAjs|L*1PmJifN)$)=%C)M$dz0<YJ%fJ5<Nh*M}2}osjJ<jAbgK#7r1Sa9+OoOChl^$T^3n(wh1K(&I7il;NXn((QbgpL&1hxyorgBd3)2){MCMeKwnb9b6gZ%jtHfkylDuG#Du`J4TCQ1OtM6@57-!@B-b6pvnlr6OTX57?=-=~KXmv;wrsTYotNa<Hmv!0Y~+k|Vj;c|myUy(*f_=E+)*H!HE|KU1%mn?BH;n)vxs-BC<Zy=OL%cLSR{Z2_(E_;!b~MsH9x)aRcw5&H8nd%dYGVRRCsXiPS_8o5F~0P0t3C{>?vT{#vyBz%~gj!H~n>tyZU(M7TWC;y%~T66hj^kY(ils3g26c1)<O~S`lVH27EGF3OMX|z-L6gFD;YrNN}mzej?H@F~XGnm|%Ss+*g1g`#n;`;wfTfaJT5@IwgmAd4=R;DBJ5A?_fJ}rVE2HwUZsBq-G3x*o21pD3JnfARSCq<YnPCj}C0tpVFveZ|%h_<F-{Jn&sslF*7Gbn`t=CT=v|^tyhR?CdP#T<dt&dYqObCTzN>GK{F-Iy41vGac`WBl7|87K4%a}2|daw6)7Z}j-+tty*-SWdrnVi(kmkiN$g0r0;Z)#i5ZNeEX;U}v;)MD2)84XE7)P@F%}lu*Wu2<hT#PjdhQ0WI5$r_npm=fKZu)ilo6N3-!$x9!c-#XP|pg)$L!TegtMCHK<SF|BA^D|{FsE99+}@;APtrT0FCyT0Ejh&JTeUNby(m@h!Tk3k)YTVHt<pdI#4%DXEdj%2o`A*y!n*l<?l((gzqnz6M!=XIO3=5d4PamY&WxeC42E^T}^g=V|CzIq$UxUODf4WVV#5%0Sm7+tKUn6jl1qKhSUfUJ+|B=7Y@)ZLM?99A-fri$lN3v8wXr=B-90Q*{1BgTC^?O$BuAItlr9j#93Y}*2ofToXCj5ER)fQ`|7vJOGLRnkB`vY$*z6INF%{z6|F^{IbkwOrnNl`_9e$j0f=S9{!Rk#132#qU1LltAX{EvP7(@Ucsr7C*6d!9IOh-+@Xb^WNOaWHv`$0?!U_UdI1*iENK<sSNzCL{63c9D3CylUraO-x)bD+^^YrTO`yZ2g=<9#@Xz#CYnAdo}ymE&mbtae9zr6c4RC(`b+y;bKT)SQS?lyJ&#xCDZ&q^5MlHWXPA-Aw$k$h>{i3>A{4~W;MG)#={QDdxve<Ystd*p5QCz7*%Z}OpsI#NcJvxOp5)1zdcCeOhjpa;xiSe&;Nz)w8E6U@`=!6dBQ>d`H1;1FMxCsiE`h*&{^rMkG)c9Ra_jzHi|*KgLDgTAg`)S=lC3Fmp2=q-&>@?;c;*;yixA&`L{W4a!-4iL6z79{*;fMRWk=a)`PIDKbJj{I39^~?Z{4U|ByD(NILG~|@%NrG6@#M~j2ASo6C1!>IC2im*TJm@+)HtU5WNJhZINLZw?rXp`G8Eb@efcP4a0ZS!Q)(sfVN9kunhDuywrYE9CPwC)zy(O<4JqAlLZAf&7y^wx;9U$S=#XlWFpz{!hN<#I}KWJElNt*epwBouEA^`=WOb`k7{)Q{2v5^yaMu4&hruD2=tYGa}u}r8&8#pl;@fjnhyzp(=?p_X7R4jYoRs?^#2(84DNjS4byI&au_(E<I@gyI-#m`iHy2t)%yx|Fct%MK!!DR^T4FHs`M(j)g)?u2&1d?z;OQ56Uo}A+Jq;Xq-k^(l+#>V)=AjULYV$@y2mF2d$zJ%wY-RH-i5P&n_|Cq>^lqH0{WefXwMu<{QCWqsGXak9m?PtOH$pdeIAj*RO7b;2;<vPlW&L#~yF0MO_fUr;f2A;aqj#pCCYdP8^bLdRs4GdSb65}U9;~@mk<7$x`AYnfxWy&{kqpknre*NA%_3L+PkA8xr<@Wdy!#O5hL0YR}kZRzribGQIF!()yK0^p;$?&5bK>Iz~u?xmkpte!dH14pL(9abz;w|{MKh&4mCPQbK4mi#ev6b=iv%tTKBdh_-LG}a>hO%fBLW3bp4f{yC&&#;(tDf_Ve&ZMdKoWIr6@!E&Ief3ML}fZz+rofue_OkKv-ae6?fQq{ZhG(Ukjehr8++Hjt^N4@?t7o13IE?KZ|pq2MSN39AmGcZHz^gEeyTVY&?W$4MWVIdn;yZW|6;I+(1!S43f#YS-wNMVk*pYkq!ngVPXA$gofOA`*pWPpS^~YC$w0Li_^wA8H6IYUo@S0kgu_u@zW4ED{rU&!gQ#K-EV7Pf=5FK^nBn|EBJ*o(fySYO+h~qAMp$gvZD$I}YRyzEml{DpL^+;6j*u&m+3|K5j&wq9r7o981RauTr)lLt#WrB$T!oyZkmD2Y?r{<<K-q)7a0n*~fGa0`IJ=SqM;smQx9hv@5W|vXnu|UI4&H+5@tun-enx$XDAXvbVZn<UUw95bCd9N}b*yCQLy)*2lK!yXRT@|WR$g1G-Tf44i_5+(E2wGPNBDAJ+lPd*j^Wwp7VPIBp&<<Um-~aAE89L$Kd;}}y>-p@))bha)WPl7;1sr^ud@F6IbuIt(yZ*YbV~vYm$8}HrrFvMK`jCR*oX%|%reblOzIE5s6D$4b!&I;qU{i$(}D&~&TMzz`v#&YpFKmbNW96;(=Y0GU--=<Noosl`g~y*3F7#LMSH750b8giRhb)jc)f~PY$ioq(kwv>CdWc5bc}VV$>pRLYOzycFTxzYxa<PB=(2$wjFNPbD0!q!(Lhuts%NZcl!X<eq}ip8@Up$tfcp|gNzb9Zhk9J(au78^gl@@ti@QL~5(X%6q(tit_e(7Z7CM~bHZD~zVraG@fd6E}nilz!dUh;ZOT#foPB2K9aMuR~qgh4h8IrBtl`Vmrsf;b$5M|*k$DXp;>bT>@Jsc&6<7f>e7N)0TR&)a)l*aYqC()MR@~>nTk>LUf1sBj7!%mRKT6DVrRC0>`8;Jjizd60#u&v8rQS%E^@r2z6(bEk%L@f0UEcyM-T((-;!f7b6T?VK72$8_Ipby)Uq;CN*mjrHYc#LiXp~J08-o}{)f`KT?BP$YVf!(8t!v*=*+g}6wW^j&a?ZgHB1t{yTtgq{w&DkqV2WGdC*RQ!x16b!CZ6nB>Tq&eiN%{a&^xzOC+zRyc>h8n)yLaA3@;(IiU%y*>`UHvd>qQ7yuigHA?dlJEFYfMs^X~4`x9Ar3^m_flU&#Gcdk-J%K7G@cGw1(0u9*h%+i7l#CQ0abJ5OVk2fc7EdOu#a=!FCE+e;X4#v+u}Tt!nH%?bX|*aY~;gBb%C4^~Pz`4x`MC;GK1OpqQ2@nCd&kSaDPm<4HLW=S*P)mjH-uxz4Y5Ln{@$jhrdmY8fQOv4cQY`49WDwd0lyG+Z$<O80ytVnVkB!jAwlZ=&|qLh&W2tIX98g_d7gzy0Yo+9EOh{=SW5rqAQe+?3c3;4uR-$X;)P|_9bafPU)qv9ti9LzWge@6#d9gyCGEHU&5dP_@spq*5O<v`2fm0|cu6G7R(v}GfwZ9`&yP0hRp5;I=#+kTF`+rY%)*1rjFz{Hv>X8od}XUq6fo^%mT<f8scd8MNlJt3KW;ZYcC;iw85VXPIY54BFBn$eHo9EL{`iwJBz6J=8TK-4WWgMbc*et|3C@OMSiq71M^a%6TP*<$8+96$}($aPc|WF$ni*vA#7A3TFMSgqRb1!Z|c8BLdpB+-rM&f>`EVsg9rg_)U&nK7>wM^d{`@`3a$W`n9%x=ke~mvp!kta11Tp-st%OpJO&ud>NzwzfqtosX3@c<uRyET1>@3UYO?7AqPY?6vtidqSaL!%QAYQpdUp3pR<pAFl4bxK_V+<>mV?;Ig_8vF?O6D!lO!dz53N;wAKOPpyMuu>+exyB+Ne@s9R4(n_h!61LIfjS|9UropXHtdS7|ai@X^*v5~SmIB}0i3r^^%xM`FT0@#jDJ!tV<g;h5XwBcu06_1cD7zB4@R7?!Y{YdU_UcG&6$LG!VRB+>>ebvh3_t*vliyy|)dG!O!1EXIxu#n5ivBk@f1?eul|!H&*dDMw3Pb>JA;RK7RDNi}Z4d}hPcur?-Q+J08sl>hva9)*YF~c8_s_4$?WVg=A1wO=j1Xw<qOL7X$#H}tX5_{zBr+n(Foqw*w_e$v=w&63HP88YVs`2z72G(HxQzJb6sqMc;-^_0zHh1v%<*HfY!i3~s3OW7fFO<1q1^!p(5^5jc3j0=hNSEOuzb;(-NKMaW7D5#eRn7!BxEI~KacWmjaqb}8;_!7oj#G;^n{7Z#cfO;5gXR2Gj9p6<z;)4yNanSVft-rBgBixr(~ko0U7_GW-hp{mp-PW0#gqbj>%J_!%f_Xs8c2}uHgoVbr3c&mEL<nll?kkx@fcg?lfFYEo*ke7JuZ}-39)E@Oau5F@cG!E(s=CgHn*#fkfe0av?!YvIMa(@y?>t@#OgIh?Iz<iGd_AQ%R!^?GtSSpM^R}msJk%FykQ@Wjt&fBu?Rg-$w$z&VbKlKES9VVF-qmgLEs=&%gMfWxTQm70r2p88$o$nk#*#M&0{0Hfw@r9R1_iAi*ZLu2~-bP)1z#^eu+IG;qgdO{RWo*ko;KP`$xrQe=*7M8qb0u=hD?_BzUTo>E#NibHHrd(FJ>!2ahjzB=r#;FFcF<gq2xdPqD-qSR-z(sX^ACSAke=%)`{2x<n%ZF2=^?9*2U$uY+I`9@d@&SrP@JM68bjF}*;l2iVV79J<lp=+~Qp{5aOi)nOQk|wv-w*4HFm0K_ZZckd`dN=bD@He%3^3FqT3!LOiG>wqU^t{J%-*E&rvq$)}k{3UL`FMpzW?Byejjtilin%C+V9m`>%+60F&$JkRHFd%dR}z~}k~&_w=J8;|MlA~D_$V<1X+UeZ4(t@&if_`Zw$S=2YYCk51jFw*xv+p(rx&b`kgX7VT)~{J5g#MNm&l%RU`0V9Yub4{NtQdIR+mX1LvcRGCVsqUF-6w=UK<51ZqI0fsA$=doP&1Hbr(2l7*d9q(Vw)O&F5b0$1S*0Y2nTpK8u9oLe7c|K0S$=u$Qn|7H_){Fog(}dBr)Yv;E1aE3`|%&M{z$6PRc;-VWa@K*Jr-)5Ve!_1Av<NBv*#?tT4M?fS<^jd}U)4Kxu790Af>x<7hUfB(DMgTL-Pes}l%8~=WG_c*vp?B0F6_w$=mC+hd!fpg$HG(T?rlQ;cd0^HdA?82Nw9wQXQli`3lI)yi&BGIra5+jg#o(%a^qTxu$lB&UNQ=>`8?F*Aij;yRr3!4vk#XYiD5wV;CFf7=d#w=E3UIc-@6i7h7ZUA6K_SQR5D3S{`Q<Y3sDFYs)Pze7G7)k*1sw)zay5N;7;hLp!8}ncw+D!Xw<OQ?mF{e)*nk9r^2yAXxNWy;m+xk0yqONt=tNwqlyitF0XZQ0r%*^cM@&ZV|{?V=7`w#24{`m6aFNt#&Ds1I#e|&3xW~}z;r@im*vnMq!oz+U#JAf2T4dDSYRbsQZk)sz*_cjnKS8+NkND8J_$w40mK`b<&Xph8_vDulaGeHC>n7wF_I{{0fwS44*7;teNo3^+7=J5?BcVfVX&ph^34VE{sM{2ON>EC9|->7BC>Pzh0<ahzPLX#<z7!;_t-j8}q$PlI~r=VpX?Z&n_RKp=+k;KTR113g+Mik*Tw*xE~d1&@zeKSwe2OH)c;|0k)?4z)QdSzur-3Y?);c$s+&lDdm%T%C*8%gLv`3A_Eg2Y`kd5BZe*)QRnK~c+oNh)pP{4bJZRlu)XZW5*)UlNOtMiHR^WdJX?%;dzu50MVPKRAC=x}xcjip!a`$pBBNvsnXvk}wO%l#(-k<JCYBEMoWuzy^dupR%Vn{v|X1g(hgY^Iv2+E}IpMPn>*1fb5m@!bF$xi}DOo+`%#O_@2#v<4Ofh$JQcL5`PcIv}^?y=p23n{$FWLEs?~Z@zCo#kH4ya{wyB;pARP^on8O)A@NVzdGg-w)3<7OuA>Kc?}9vuI1A!~LBt2Urok8V<E!-tUqM9B)$8Qk7wv?5kKWpO_MNf3NmpQI5N{xT5R5Y~i2L<;ZCtWIVuYi92;hLcI~AoFU^ox$pt}jbzGtg=5p{skvm?C<!HK`51K4s2>;m5<sYeAr0*iC;(VYF{^w~UZMqhy2`$a2j2CmV#73#0c)FFvZJ;w%(@@kbTO<JNh`wnu3gNSt#Nk2Fm!~zqJ1RDb}Pe(1Rc%Vb{9s)j3ZQKs{!8M#Tnog8f#~%Pd7p${+_!%S}wn^wN?HCut?mq)IPZ3;*sGPXV%pCxvfZbH<v=y3;3s%Z+(`CU7EwG#wEPtczC8q1k)R_pgKK$IVp_`r`u2U)3Nd?z;Ik5>Q^!SEa7Q(iJG72S*KhQMvJCBJF7sb~OV-_i);vikPfr@-z25cEOJa^!Sq$yd*7qID^Lm-BJ=g9-iTrg9~mvo}8L;6La6>L~I8J|x~%+63tK0YbY8_9vqj-Q>m78#FC)>%_2=-`np>uFUb+YUsQX260F5%IZ!HGEb?<4vnMdR!2zp)Pq1k-jS9%=%bPBQ}-U*`zcRn~vLYXt_yXCe_c0Go|uwVqqeQhB+0Rp{5_tTNB7skD^yI#quab6w$+8sx=v529UJq>7VMK-Xj+_*v=j7Lh^vqZH6pO_VgLDE`@B@{$LKEkd|#suJ+>U%ll7v9zUr+_yTR;evyX~Y-qD_cH*0Zix58BC||whHbeYbAOpR0aw0Kt3~6s8@skt7@r1vH<n^_C-#{X$-+xp4!}Fb|U)DeRx%TWf(E@lNi4X$k<){X}jHPz<`<*BMRl9kg877vLY#v8&b0&VOB9;IWx@ZlBP#fN`UHyegj$=Ty%Va8%?)>tTPj~<J1oa0E!r1T4wEKI1eS7!r2lWrX-h1>0lAcUYXy`$64FRrnNYiW7uybzFpc7u2n}STU*+6DLjov)LTrj~<ojsz0&pBvt4MPUh0h@inJP2^OBbHfdzl_AeQbsPz*u-iEvg=Z3D1>z$kt**--l1qWD5-2FqG*qcUD?jKT#*s5XI-cedkiTOwlrfhT~TJ{1Fm}pO|LYgnDvC{&*svMx>u_m={U7>W3-b~3px&ZUPL@L=`%RfN;F{DWX4Ty9Xm5M8ylf{Zb>3sUIyHNlf`>CfIZgG?h$B;coO%cN0*pzqBS;yF~^qQ$ZW-0P14R>zmD`&B7c)=Wa%Yoll?o-@9kZ=3b(+0^+E0T-%}&lk3S;8M<drTC(3eGnV9g~vyLP6qkjDZLjgf7ARZ!V1#mj=+SUJl`QZawc5QIlC=&-v#EDmPYAQ&E#P00|)da92F0Cn>gja|`?8~OWZ7i$G21Nn`-kIOub9=Mxgis6kA<o`!_Rzl!{u<`bRzko&{F-t|yaOWDt9FN|jW?|OihleN=~@jtfOR`M(WDdh%+CGXV;Q^*u?~cNK_sFBwRlIx&eJcz5Ow<xyYGFr^WtIsy`Ruj?%sd6_wY;f6RzF7bBkOT$kDT~GuKVR`%%r>?T^UF{hrqum?u+KM~Tg;u?GkBgF#q|@-P+f6_j;X7HqyF3q=|N)8DL<PTDa-t_;G6n%@E@yNg;B+$du@n<thTifat5XAAeG5%Gg%m+ScF9vpbHO7rotiA2&0xgee}P4PnbrZZzar+3B#aU6+W{^hORhaWEckX()gTF5(c=jP5&@1UR4@B<quOq&Iw$?pI8uq)Kp<1ku~U^Jc+ZA+pb$A<v~_>F++(4#;qdpPM5s8$gcx!!Wr=a+QMw}qi>zm5j5?A1A&Oica!P3@ZF=|9lOd8w5#n|2GVlk&#i+KF__C)1XRPhT^{05ubBco=QpWLyWa%^io0{Td$vvwrXFtz{%w#JHP@ll+oiei4pzdL$6D&6<H(&aAaEF`o1Bi)a!1!+3j}@qEd69(jE7#x}xNB5xP6sNLw7Q#f-J)>3&61B)j>{B(ur5a&v3maNmCsPV-XqRAVr5pXs<J3TiQPm%<zMs53n8zM=BLA?K41<V6A9KuZ81Lk7$$q6KA&dy83E6;u>rc<dlSS)DapM18=9Afc2Gn*TaekGRiZDjKcbIG`Om>%5n8bw;>W5`8-RboTA+<p20i%p_aN97u_+{tM0F}NWWf=Y><<caoPLJI=Bf-yI$I+g&`U~Y(n|Nb949wz7'
_RUNTIME_SHA = '1f6e4d72c7c1bd808acc62f8ae3c5f1aad1a69289197dec9c802865b5518d840'
_runtime_bytes = zlib.decompress(base64.b85decode(_RUNTIME_PAYLOAD))
if hashlib.sha256(_runtime_bytes).hexdigest() != _RUNTIME_SHA:
    raise RuntimeError('封存源运行时校验失败')
exec(compile(_runtime_bytes, '<sealed_source_and_strictE1>', 'exec'), globals())
del _runtime_bytes

"""本段由交付构建器与只读源环境运行时合成一个独立文件。"""
import bisect
from collections import defaultdict

EFFECT_VERSION = 'E01_S_AC_198M_20261027_V1'
EFFECT_ROOT = 'uam_E01_S_AC_198m_v1'
ARMS = ('R_SOURCE', 'R_SHARED', 'R00_NOALIGN', 'R10_S', 'R01_AC', 'R11_SAC')
ROLLOUT = 20_480
EFFECT_STEPS = 49 * ROLLOUT
EFFECT_CAP = 200_000_000
CHECKPOINT_STEPS = tuple(math.ceil(49 * i / 21) * ROLLOUT for i in range(1, 22))
EFFECT_RUNTIME_PID = None
_effect_old_initialize = initialize_full_runtime
_effect_old_train = full_train
_effect_old_resume = full_resume_hook


def effect_registry():
    """198run先锁定全部对照，确认组不按筛选赢家改变。"""
    rng = random.Random(20261027)
    phases = [('P1_MAIN_CORE', [('E1', 3, c) for c in ('LQ', 'J3')], (1201, 1202, 1203)),
              ('P2_CONFIRM_CORE', [('E1', 3, c) for c in ('LQ', 'J3')], (1301, 1302, 1303)),
              ('P3_MAIN_SCALE', [('E1', k, c) for k in (2, 4) for c in ('LQ', 'J3')], (1201, 1202, 1203)),
              ('P4_MAIN_SOURCE', [('E0', k, 'LQ') for k in (2, 3, 4)], (1201, 1202, 1203))]
    tasks = []
    for phase, conditions, seeds in phases:
        for environment, k, control in conditions:
            for seed in seeds:
                order = list(ARMS); rng.shuffle(order)
                for arm in order:
                    method = ('J3__' if control == 'J3' else 'SINGLE__') + arm
                    tasks.append(dict(id=f'{phase}__{environment}__K{k}__{control}__{arm}__s{seed}',
                        environment=environment, k=k, fleet=12 if environment == 'E1' else None,
                        control=control, arm=arm, method=method, seed=seed, stage=phase,
                        block=f'{phase}__{environment}__K{k}__{control}__s{seed}', run_order=len(tasks) + 1))
    assert len(tasks) == 198 and len({t['id'] for t in tasks}) == 198
    assert len(CHECKPOINT_STEPS) == len(set(CHECKPOINT_STEPS)) == 21
    assert len(tasks) * EFFECT_STEPS == 198_696_960 < EFFECT_CAP
    return tasks


class EffectSchema:
    """有限候选查询表是五臂共同输入，不增加原始事件槽和隐形信息权限。"""
    def __init__(self, k):
        self.k = int(k); self.frame = 4 + 8 * self.k; self.history = 6 * self.frame
        self.queries = self.k + 2
        offset = self.history
        self.od = slice(offset, offset + 4); offset += 4
        # 两种角色、K个候选、K+2锚点、三张对照表、全部K站、八项摘要。
        self.table_shape = (2, self.k, self.queries, 3, self.k, 8)
        self.table = slice(offset, offset + math.prod(self.table_shape)); offset = self.table.stop
        self.anchors = slice(offset, offset + 2 * self.k); offset = self.anchors.stop
        self.air_valid = slice(offset, offset + self.k); offset = self.air_valid.stop
        self.passenger_active = offset; self.pack = offset + 1
        self.return_mask = slice(2 * self.pack, 2 * self.pack + self.k)
        self.dim = 2 * self.pack + self.k

    def manifest(self):
        return dict(k=self.k, frame=self.frame, history=self.history, pack=self.pack, dim=self.dim,
            table_shape=self.table_shape, common_information='五增强臂相同的全部候选时间查询表及基准/候选/参考输入',
            no_alignment='所有臂能读相同查询表；S只显式选自己的列，非S仍能学习利用全部列',
            ac='显式差分是可逆信息重组，不等同长期因果信用',
            control='乘客/critic前态；飞机actor仅真实dispatch后态；不增加时钟/权限')


def arm_of(method):
    arm = method.split('__', 1)[-1]
    if arm not in ARMS:
        # 仅规则回放会用旧方法名，不能借此启用新的物理权限。
        return 'R_SOURCE'
    return arm


def _empty_plan(scenario, k, priority):
    """只读复算既有执行器将如何处理全部eligible飞机；不改任何真实状态。"""
    from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
    vids = tuple(str(v) for v in CANDIDATES[k])
    queue = {v: len(scenario.vertiports.vertiport_list[v].person_list) for v in vids}
    ready = {v: sum(e.state.name == 'IDLE' and not e.passenger_ids
                    for e in scenario.vertiports.evtols_at_vertiport[v]) for v in vids}
    available = sorted((e for e in scenario.vertiports.evtols_at_vertiport.get('2', [])
                        if e.state.name == 'IDLE' and not e.passenger_ids), key=lambda e: str(e.id))
    plan = []
    for plane in available:
        requested = vids[priority] if priority is not None else None
        target = requested if requested is not None and queue[requested] > 0 else max(
            vids, key=lambda v: (queue[v], -ready[v], -int(v)))
        if queue[target] <= 0:
            break
        stations = scenario.vertiports.vertiport_list
        distance = scenario.vertiports._calculate_distance(
            stations[str(plane.current_vertiport_id)].vertiport_position, stations[target].vertiport_position)
        energy = distance * plane.spec.energy_consumption_kwh_per_km
        if plane.battery_kwh < energy:
            continue
        flight = distance / plane.spec.max_speed * 60.
        charge = max(0., plane.spec.battery_capacity_kwh - (plane.battery_kwh - energy)) / VERTIPORT_CHARGE_RATE[target]
        # 这是沿源分钟离散化的描述符，不将一tick近似说成精确未来队列。
        eta = math.ceil(max(0., flight)) + math.ceil(max(0., charge))
        plan.append(dict(aircraft=str(plane.id), target=target, eta=float(eta), flight=float(flight)))
        queue[target] = max(0, queue[target] - int(plane.spec.capacity))
    return plan


def _committed(scenario, k):
    """只收已经选站在途的人和已承诺空机；不读取未reveal请求来构造观测。"""
    from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
    output = []
    for vid in CANDIDATES[k]:
        key = str(vid)
        incoming = sorted(max(0., float(p.current_timer or 0.)) for p in scenario.persons.persons.values()
            if str(getattr(p, 'origin_vertiport_id', '')) == key and getattr(p, 'state', '') == 'enroute'
            and getattr(p, 'sub_state', '') == 'to_vertiport')
        ready = []; idle = 0
        for plane in scenario._all_evtols.values():
            if plane.passenger_ids:
                continue
            state = plane.state.name
            here = str(plane.current_vertiport_id) == key and state != 'FLYING'
            arriving = state == 'FLYING' and str(plane.target_vertiport_id) == key
            if here and state == 'IDLE':
                idle += 1
            elif (here and state == 'CHARGING') or arriving:
                rate = VERTIPORT_CHARGE_RATE[key] if arriving else plane.spec.charge_rate_kwh_per_min
                charge = math.ceil(max(0., plane.spec.battery_capacity_kwh - plane.battery_kwh) / max(rate, 1e-8))
                flight = math.ceil(max(0., float(plane.remaining_time))) if arriving else 0
                ready.append(float(flight + charge))
        output.append(dict(queue=len(scenario.vertiports.vertiport_list[key].person_list), incoming=incoming,
                           idle=idle, ready=sorted(ready)))
    return output


def _summary8(background, anchor, demand_eta=None, supply_eta=()):
    """累计承诺摘要而非精确未来队列；不扣除未决定的未来服务。"""
    arrivals = bisect.bisect_right(background['incoming'], anchor)
    arrivals += int(demand_eta is not None and demand_eta <= anchor)
    committed = background['queue'] + arrivals
    supply = sorted(background['ready'] + list(supply_eta))
    other = bisect.bisect_right(supply, anchor)
    next_gap = min(supply, default=600.) - anchor
    return [background['queue'] / 300., arrivals / 300., (committed // 4) / 75.,
            (committed % 4) / 4., background['idle'] / 20., other / 20., next_gap / 600., anchor / 600.]


def query_tables(scenario, k, focal, passenger_stage, joint):
    """同一张完整小表供所有增强臂使用，S/AC没有额外的隐藏输入。"""
    background = _committed(scenario, k)
    stations = scenario.vertiports.vertiport_list
    p_anchor = [max(0., float(scenario.vehicles.estimate_travel_time(
        focal.origin_position, stations[str(v)].vertiport_position))) if focal is not None else 0.
        for v in CANDIDATES[k]]
    plans = [_empty_plan(scenario, k, i) if joint else [] for i in range(k)]
    reference = _empty_plan(scenario, k, None) if joint else []
    a_anchor = [min((e['eta'] for e in plans[i]), default=600.) for i in range(k)]
    table = np.zeros((2, k, k + 2, 3, k, 8), np.float32)
    for role, anchors in enumerate((p_anchor, a_anchor)):
        queries = list(anchors) + [float(statistics.median(anchors)), 0.]
        for candidate in range(k):
            for j, anchor in enumerate(queries):
                # 每个候选保留所有站的变化，不能只给受益站而漏掉被挤占站的机会成本。
                for station_index, vid in enumerate(CANDIDATES[k]):
                    key = str(vid)
                    add_supply = [e['eta'] for e in plans[candidate] if e['target'] == key] if role else []
                    ref_supply = [e['eta'] for e in reference if e['target'] == key] if role else []
                    base = _summary8(background[station_index], anchor)
                    plus = _summary8(background[station_index], anchor,
                        demand_eta=p_anchor[candidate] if not role and focal is not None and passenger_stage and station_index == candidate else None,
                        supply_eta=add_supply)
                    ref = _summary8(background[station_index], anchor, supply_eta=ref_supply)
                    table[role, candidate, j, :, station_index] = (base, plus, ref)
    return table, np.asarray(p_anchor + a_anchor, np.float32) / 600., np.asarray([bool(p) for p in plans], np.float32)


def effect_initialize():
    """只替换观测和小残差；物理分钟、奖励、PPO目标及执行器沿用封存版本。"""
    global EFFECT_RUNTIME_PID, FullSchema, FullEnv, FullFeatures, FullPolicy, FullPPO, FullVecNormalize, FullCheckpoint
    if EFFECT_RUNTIME_PID == os.getpid():
        return
    _effect_old_initialize()
    OldEnv, OldPolicy, OldPPO = FullEnv, FullPolicy, FullPPO
    from torch import nn
    import torch.nn.functional as F
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3 import PPO

    class EffectEnv(OldEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.arm = arm_of(self.method)
            self._timing = defaultdict(float)

        def _events_and_pack(self, history, focal=None):
            started = time.process_time(); s, d = self.scenario, self.schema
            if focal is None and s.waiting_decisions:
                focal = s.persons.persons[s.waiting_decisions[0]]
            od = list(focal.origin_position) + list(focal.destination_position) if focal else [0.] * 4
            if arm_of(self.method) == 'R_SOURCE':
                table = np.zeros(d.table_shape, np.float32); anchors = np.zeros(2 * self.k, np.float32); air_valid = np.zeros(self.k, np.float32)
            else:
                table, anchors, air_valid = query_tables(s, self.k, focal, self._generator is None, self.joint)
            pack = np.concatenate([np.asarray(history, np.float32).reshape(-1), np.asarray(od, np.float32) / 60.,
                                   table.flatten(), anchors, air_valid, [float(bool(s.waiting_decisions))]])
            self._timing['event_pack_cpu_seconds'] += time.process_time() - started
            if pack.shape != (d.pack,) or not np.isfinite(pack).all():
                raise AssertionError('观测长度或有限性错误，不静默补零')
            return pack.astype(np.float32)

        def prepare_step(self, actions):
            start = time.process_time(); before = self._timing['event_pack_cpu_seconds']
            value = super().prepare_step(actions)
            self._timing['physical_prepare_cpu_seconds'] += time.process_time() - start - (self._timing['event_pack_cpu_seconds'] - before)
            return value

        def finish_step(self, actions):
            start = time.process_time(); before = self._timing['event_pack_cpu_seconds']
            value = super().finish_step(actions)
            self._timing['physical_finish_cpu_seconds'] += time.process_time() - start - (self._timing['event_pack_cpu_seconds'] - before)
            self._timing['physical_ticks'] += 1
            return value

        def consume_timing(self):
            result = dict(self._timing); self._timing.clear(); return result

        def counterfactual_pre(self, od):
            raise RuntimeError('本轮不使用破坏物理一致性的输入置换探针')

    class EffectNormalize(VecNormalize):
        def normalize_obs(self, obs):
            value = super().normalize_obs(obs)
            if not hasattr(self, '_effect_schema'):
                self._effect_schema = EffectSchema(int(self.venv.get_attr('k', indices=[0])[0]))
            d = self._effect_schema
            for start in (0, d.pack):
                value[..., start + d.history:start + d.pack] = np.asarray(obs)[..., start + d.history:start + d.pack]
            value[..., d.return_mask] = np.asarray(obs)[..., d.return_mask]
            return value.astype(np.float32)

    class EffectFeatures(BaseFeaturesExtractor):
        def __init__(self, observation_space, k=2, method='SINGLE__R_SOURCE', environment='E0'):
            super().__init__(observation_space, 128)
            from utilss.encoding import TemporalLSTMExtractor
            self.k, self.method, self.environment = k, method, environment
            self.arm = arm_of(method); self.schema = d = EffectSchema(k)
            self.temporal = TemporalLSTMExtractor(gym.spaces.Box(-np.inf, np.inf, (6, d.frame), np.float32), features_dim=128)
            if self.arm != 'R_SOURCE':
                # 附加模块独立初始化，避免改变源网络的初始化随机序列。
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(913107 + k)
                    flat = d.table.stop - d.table.start + 3 * k + 4
                    self.context = nn.Sequential(nn.Linear(flat, 64), nn.Tanh(), nn.Linear(64, 128))
                    self.candidate = nn.Sequential(nn.Linear(128 + 16 * k + k + 1, 64), nn.Tanh(), nn.Linear(64, 1))
                    nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
                    nn.init.zeros_(self.candidate[-1].weight); nn.init.zeros_(self.candidate[-1].bias)

        def auxiliary_parameters(self):
            return []

        def encode(self, pack):
            d = self.schema
            value = self.temporal(pack[:, :d.history].reshape(-1, 6, d.frame))
            return value if self.arm == 'R_SOURCE' else value + self.context(pack[:, d.od.start:d.passenger_active])

        def forward(self, obs):
            return self.encode(obs[:, :self.schema.pack])

        def dispatch_features(self, obs):
            return self.encode(obs[:, self.schema.pack:2 * self.schema.pack])

        def correction(self, pack, role):
            if self.arm == 'R_SOURCE':
                return torch.zeros((len(pack), self.k), device=pack.device, dtype=pack.dtype)
            d = self.schema; b = len(pack)
            values = pack[:, d.table].reshape(b, *d.table_shape)[:, role]
            if self.arm == 'R_SHARED':
                first = values[:, :, :, 0].mean((1, 2)).flatten(1)[:, None].expand(-1, self.k, -1)
                second = values[:, :, :, 1].mean((1, 2)).flatten(1)[:, None].expand(-1, self.k, -1)
            else:
                index = torch.arange(self.k, device=pack.device) if self.arm in ('R10_S', 'R11_SAC') else torch.full((self.k,), self.k, device=pack.device, dtype=torch.long)
                selected = values[:, torch.arange(self.k, device=pack.device), index]
                first, second = selected[:, :, 0].flatten(2), selected[:, :, 1].flatten(2)
                if self.arm in ('R01_AC', 'R11_SAC'):
                    # 飞机差分相对合法LQ参考；乘客差分相对尚未加入焦点承诺的背景。
                    second = second - (selected[:, :, 2].flatten(2) if role else first)
            common = self.encode(pack)[:, None].expand(-1, self.k, -1)
            ids = torch.eye(self.k, device=pack.device)[None].expand(b, -1, -1)
            anchor = pack[:, d.anchors].reshape(b, 2, self.k)[:, role, :, None]
            return self.candidate(torch.cat([common, first, second, ids, anchor], -1)).squeeze(-1)

    class EffectPolicy(OldPolicy):
        def _build(self, lr_schedule):
            ext = self.features_extractor
            extras = {name: ext._modules.pop(name) for name in ('context', 'candidate') if name in ext._modules}
            try:
                super()._build(lr_schedule)
            finally:
                for name, module in extras.items():
                    ext.add_module(name, module)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        def passenger_parts(self, obs):
            logits, values = super().passenger_parts(obs)
            ext = self.features_extractor
            correction = ext.correction(obs[:, :ext.schema.pack], 0)
            active = obs[:, ext.schema.passenger_active] > .5
            return logits + correction * active[:, None], values

        def return_distribution(self, obs, passenger):
            ext = self.features_extractor; d = ext.schema
            dispatch = ext.dispatch_features(obs)
            ids = F.one_hot(passenger.long(), self.k).to(dispatch.dtype)
            logits = self.return_head(torch.cat([dispatch, ids], -1))
            logits += ext.correction(obs[:, d.pack:2 * d.pack], 1)
            return torch.distributions.Categorical(logits=logits.masked_fill(obs[:, d.return_mask] < .5, -1e9))

        def get_distribution(self, obs):
            raise RuntimeError('两阶段策略需先sample_passenger，再由真实prepare_step产生后态后sample_return；不可用普通predict伪造后态')

        def forward(self, obs, deterministic=False):
            raise RuntimeError('本轮仅使用显式两阶段采样接口，不能将未发生的飞机后态用于乘客动作')

    class EffectPPO(OldPPO):
        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
            if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
            if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)
            self._effect_timing = getattr(self, '_effect_timing', [])
            self._effect_timing.append(dict(step=self.num_timesteps, rollout_wall_seconds=time.perf_counter() - start,
                environment_cpu_sums=env.venv.env_method('consume_timing'),
                note='rollout墙钟含推理/IPC/环境；子环境CPU时间不可直接与墙钟相加；physical字段仍含源状态编码，不是纯物理内核'))
            return result

        def train(self):
            if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            PPO.train(self)
            if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)
            self._effect_timing[-1]['ppo_update_wall_seconds'] = time.perf_counter() - start
            if self._full_save_hook is not None:
                self._full_save_hook(self)

    class EffectCheckpoint(BaseCallback):
        def __init__(self, directory): super().__init__()
        def _on_step(self): return True

    FullSchema, FullEnv, FullFeatures = EffectSchema, EffectEnv, EffectFeatures
    FullPolicy, FullPPO, FullVecNormalize, FullCheckpoint = EffectPolicy, EffectPPO, EffectNormalize, EffectCheckpoint
    for cls in (EffectEnv, EffectFeatures, EffectPolicy, EffectPPO, EffectNormalize, EffectCheckpoint):
        cls.__qualname__ = cls.__name__; cls.__module__ = __name__
        globals()[cls.__name__] = cls
    EFFECT_RUNTIME_PID = os.getpid()


initialize_full_runtime = effect_initialize


def effect_budget(root):
    """中断和失败同样计费，未落盘的一段按完整rollout上界计。"""
    measured = uncertain = 0
    for path in Path(root).glob('cells/*/attempts/*.json'):
        item = read_json(path, {})
        measured += item.get('measured_transitions', 0)
        if item.get('status') != 'FINISHED': uncertain += item.get('unlogged_tail_upper_bound', ROLLOUT)
    return measured, uncertain


def effect_train(task, root, config):
    global full_resume_hook
    folder = Path(root) / 'cells' / task['id']
    pointer = read_json(folder / 'RESUME_POINTER.json', {}) if config['resume'] else {}
    initial = int(pointer.get('step', 0)); used, uncertain = effect_budget(root)
    if used + uncertain + max(0, EFFECT_STEPS - initial) > EFFECT_CAP:
        raise RuntimeError('失败/中断计入后的预算上界超过2亿，拒绝超支')
    attempt = folder / 'attempts' / f'{time.time_ns()}.json'
    save_json(attempt, dict(start_step=initial, measured_transitions=0, status='RUNNING', unlogged_tail_upper_bound=ROLLOUT, time=now()))
    original = full_resume_hook
    def wrapped(directory):
        old = _effect_old_resume(directory)
        def hook(model):
            start = time.perf_counter(); old(model)
            if model.num_timesteps in CHECKPOINT_STEPS:
                full_save_pair(model, Path(directory) / 'checkpoints', model.num_timesteps)
            rows = getattr(model, '_effect_timing', [])
            if rows:
                rows[-1]['save_wall_seconds'] = time.perf_counter() - start
                save_csv(Path(directory) / 'TIMING.csv', rows)
            manifest=read_json(Path(directory)/'MODEL_MANIFEST.json',{})
            manifest.update(arm=task['arm'],auxiliary='全部关闭，无额外优化',runtime_sha256=_RUNTIME_SHA,
                information='所有增强臂相同候选查询表；S与显式差分只重组信息',checkpoint_contract=list(CHECKPOINT_STEPS))
            save_json(Path(directory)/'MODEL_MANIFEST.json',manifest)
            save_json(attempt, dict(start_step=initial, last_logged_step=model.num_timesteps,
                measured_transitions=max(0, model.num_timesteps - initial), unlogged_tail_upper_bound=ROLLOUT,
                status='RUNNING', time=now()))
        return hook
    full_resume_hook = wrapped
    try:
        _effect_old_train(task, root, config)
        manifest = read_json(folder / 'MODEL_MANIFEST.json', {})
        manifest.update(arm=task['arm'], auxiliary='全部关闭，无额外优化', runtime_sha256=_RUNTIME_SHA,
            information='所有增强臂相同查询表；显式S/差分只是结构重组',
            checkpoint_contract=list(CHECKPOINT_STEPS), source_is_external_reference=True)
        save_json(folder / 'MODEL_MANIFEST.json', manifest)
    finally:
        full_resume_hook = original
        record = read_json(attempt, {})
        record.update(status='FINISHED' if (folder / 'TRAIN_END.json').exists() else 'INTERRUPTED', time=now())
        save_json(attempt, record)


def effect_setup(task, root):
    return full_setup(task, root)


def _lq_priority(env, pending):
    """J3规则也只能给每分钟一个优先级，不借原LQ每架重算偷增控制自由度。"""
    valid = [i for i in range(env.k) if pending[env.schema.return_mask][i] > .5]
    def key(i):
        vid = str(CANDIDATES[env.k][i])
        ready = sum(e.state.name == 'IDLE' and not e.passenger_ids for e in env.scenario.vertiports.evtols_at_vertiport[vid])
        return len(env.scenario.vertiports.vertiport_list[vid].person_list), -ready, -int(vid)
    return max(valid, key=key)


def effect_replay(model, norm, root, task, config, trace, seed, rule=None, capture=False):
    """一次回放同时保留完成者ATT、完成率、人时和实际控制；不修复源奖励。"""
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    os.environ['UAM_STRICT_LOG_EVENTS'] = '1' if capture else '0'
    env = FullEnv(root, task['k'], config['eval_horizon'], trace, task['method'], task['environment'])
    obs, _ = env.reset(seed=seed)
    author = AuthorFunctions(rule) if rule else None
    counts = [0] * task['k']; area = 0.; exposure = defaultdict(float); ticks = []
    start = time.perf_counter()
    try:
        while True:
            s = env.scenario; finished = set(s.finished_ids)
            area += sum(pid not in finished and float(getattr(p, 'spawn_time', s.time)) <= s.time for pid, p in s.persons.persons.items())
            active = bool(s.waiting_decisions)
            for vid in CANDIDATES[env.k]:
                key = str(vid); q = len(s.vertiports.vertiport_list[key].person_list)
                available = sum(e.state.name == 'IDLE' and not e.passenger_ids for e in s.vertiports.evtols_at_vertiport[key])
                category = ('both_ready' if q >= 4 and available else 'batch_ready_aircraft_missing' if q >= 4
                    else 'batch_missing_aircraft_ready' if available else 'both_missing')
                exposure[category] += q
            audit=None
            if capture:
                committed=_committed(s,task['k'])
                audit=[dict(queue=x['queue'],idle=x['idle'],incoming_count=len(x['incoming']),
                    next_incoming=min(x['incoming'],default=None),committed_ready_count=len(x['ready']),
                    next_committed_ready=min(x['ready'],default=None)) for x in committed]
                source_frame=obs[env.schema.history-env.schema.frame:env.schema.history].tolist()
            pre = norm.normalize_obs(obs[None]) if norm is not None else obs[None]
            if rule:
                passenger, _ = adapter_decision(s, task['k'], author, np)
            else:
                with torch.no_grad():
                    chosen, _, _ = model.policy.sample_passenger(torch.as_tensor(pre, device=model.device), True)
                passenger = int(chosen[0])
            if active: counts[passenger] += 1
            pending_raw = env.prepare_step(passenger)
            if rule:
                returning = _lq_priority(env, pending_raw) if env.joint else -1
            else:
                pending = norm.normalize_obs(pending_raw[None])
                pending[:, :env.schema.pack] = pre[:, :env.schema.pack]
                with torch.no_grad():
                    returning = int(model.policy.sample_return(torch.as_tensor(pending, device=model.device),
                        torch.as_tensor([passenger], device=model.device), True)[0])
            if capture:
                ticks.append(dict(time=env.scenario.time, passenger_active=active, passenger=passenger,
                    requested_return=returning if env.joint else None,
                    pre_committed_station_summary=audit,source_current_frame=source_frame,
                    return_mask=pending_raw[env.schema.return_mask].tolist(),
                    aircraft_valid=pending_raw[env.schema.pack + env.schema.air_valid.start:env.schema.pack + env.schema.air_valid.stop].tolist()))
            obs, reward, done, info = env.finish_step(returning)
            if done: break
        result = _base_episode_metrics(env, counts, area)
        result.update(eval_seed=seed, trace_sha256=digest(trace), wall_seconds=time.perf_counter() - start,
            control=task['control'], rule=rule, control_diagnostics=env.control_stats,
            control_trace=env.control_trace, decision_trace=ticks,
            tick_snapshot_waiting_exposure=dict(exposure),
            exposure_note='分钟起点的队列人数暴露，不是假称精确分解每人的等待因果时长',
            descriptor_note='ceil飞行＋ceil充电为承诺就绪描述符，可能有一tick边界误差；未决服务不预测成真值')
        if not capture:
            result['aircraft_events'] = []; result['resource_timeline'] = []
        return result
    finally:
        env.close()


def _effect_validation(config):
    # 默认仍是固定源训练trace的两个随机回放，不冒充两条独立需求。
    traces = config.get('validation_traces') or [config['trace']]
    return [(i, path, seed) for i, path in enumerate(traces) for seed in config['eval_seeds']]


def effect_eval(task, root, config, testing=False):
    directory = effect_setup(task, root)
    import pickle
    scope = 'test' if testing else 'evaluation'
    start = time.perf_counter()
    if testing:
        selection = read_json(directory / 'BEST_SELECTION.json')
        if not selection or not config.get('test_traces'):
            raise RuntimeError('独立测试需要已锁定Best和明确的test-traces，不能编造测试集')
        wanted = [selection['step']]
        bank = [(i, p, s) for i, p in enumerate(config['test_traces']) for s in config['eval_seeds']]
    else:
        wanted = CHECKPOINT_STEPS; bank = _effect_validation(config)
    curves = []
    for step in wanted:
        cp = read_json(directory / 'checkpoints' / f'step_{step:09d}.json')
        if not cp: raise RuntimeError(f'缺少完整rollout后的检查点：{step}')
        model = norm = None; episodes = []
        try:
            for index, trace, seed in bank:
                ep_path = directory / scope / 'episodes' / f'step_{step:09d}__trace{index:02d}__s{seed}.json'
                result = read_json(ep_path) if config['resume'] else None
                if result is None:
                    if model is None:
                        model = FullPPO.load(directory / 'checkpoints' / Path(cp['model']).name, device=config['eval_device'])
                        with (directory / 'checkpoints' / Path(cp['vecnorm']).name).open('rb') as stream: norm = pickle.load(stream)
                        norm.training = False; norm.norm_reward = False
                        if norm.observation_space.shape != (EffectSchema(task['k']).dim,):
                            raise RuntimeError('模型与归一化schema不匹配')
                    result = effect_replay(model, norm, root, task, config, trace, seed,
                        capture=testing or step in (CHECKPOINT_STEPS[0],CHECKPOINT_STEPS[len(CHECKPOINT_STEPS)//2],EFFECT_STEPS))
                    save_json(ep_path, result)
                episodes.append(result)
            row = dict(step=step, co=mean([e['ATT_completed_only'] for e in episodes]),
                completion=mean([e['completion'] for e in episodes]), rpt=mean([e['restricted_person_time_per_generated'] for e in episodes]),
                all_ATT=mean([e['ATT'] for e in episodes]) if all(e['full'] for e in episodes) else None,
                episodes=len(episodes), simulation_minutes=mean([e['simulation_minutes'] for e in episodes]))
            if row['co'] is None: raise RuntimeError('本检查点无完成者ATT，不用0填补')
            save_json(directory / scope / f'step_{step:09d}.json', row); curves.append(row)
            print('[回放]', task['id'], step, row['co'], row['completion'], flush=True)
        finally:
            del model, norm
    if testing:
        save_json(directory / 'TEST_END.json', dict(task=task, result=curves[0], time=now()))
        return
    best = min(curves, key=lambda r: (r['co'], r['step']))
    selection = dict(step=best['step'], metric='验证Best完成者ATT；不是全体ATT',
        validation_hashes=[digest(p) for p in config.get('validation_traces') or [config['trace']]],
        trace_scope=config['validation_scope'], completion=best['completion'], rpt=best['rpt'], time=now())
    previous = read_json(directory / 'BEST_SELECTION.json')
    if previous and previous['step'] != best['step']:
        raise RuntimeError('已有Best锁与当前评估不一致，拒绝覆盖')
    if previous is None:save_json(directory / 'BEST_SELECTION.json', selection)
    late = [r for r in curves if r['step'] >= .8 * EFFECT_STEPS]
    result = dict(task=task, best_co=best['co'], best_step=best['step'], best_completion=best['completion'],
        best_rpt=best['rpt'], late_co=mean([r['co'] for r in late]), late_sd=sd([r['co'] for r in late]),
        final_co=curves[-1]['co'], best_to_late=mean([r['co'] for r in late]) - best['co'],
        late_steps=[r['step'] for r in late], eval_seconds=time.perf_counter() - start,
        scope=config['validation_scope'], user_supplied_validation=bool(config.get('validation_traces')),
        statistical_independence_certified=False,time=now())
    save_csv(directory / 'curve.csv', curves); save_json(directory / 'CELL_SUMMARY.json', result)
    save_json(directory / 'EVAL_END.json', dict(status='EVALUATED', time=now()))


def effect_rules(task, root, config):
    directory = effect_setup(task, root); rows = []
    for rule in ('SPF', 'STTF', 'QTTI'):
        for i, trace, seed in _effect_validation(config):
            path = directory / 'episodes' / f'{rule}__trace{i:02d}__s{seed}.json'
            item = read_json(path) if config['resume'] else None
            if item is None:
                item = effect_replay(None, None, root, task, config, trace, seed, rule=rule, capture=True)
                save_json(path, item)
            rows.append(item)
    save_json(directory / 'RULES_END.json', dict(task=task, episodes=rows,
        adapters=[AuthorFunctions(r).manifest() for r in ('SPF', 'STTF', 'QTTI')], time=now()))


def effect_aggregate(root, tasks):
    status = []; results = []; rule_rows = []
    for task in tasks:
        d = Path(root) / 'cells' / task['id']
        trained = read_json(d / 'TRAIN_END.json'); result = read_json(d / 'CELL_SUMMARY.json')
        failures = [p for p in d.glob('FAILED_*.json') if '_RESOLVED_' not in p.name]
        live = read_json(d / 'LIVE.json', {})
        status.append(dict(**task, trained=bool(trained), evaluated=bool(result), failed=bool(failures),
            steps=trained['steps'] if trained else live.get('steps', 0)))
        if result:
            results.append(dict(**task, **{k:v for k,v in result.items() if k not in ('task',)}))
    for path in Path(root).glob('cells/RULES__*/RULES_END.json'):
        record = read_json(path); t = record['task']
        for rule in ('SPF', 'STTF', 'QTTI'):
            episodes = [e for e in record['episodes'] if e['rule'] == rule]
            rule_rows.append(dict(environment=t['environment'], k=t['k'], control=t['control'], rule=rule,
                co=mean([e['ATT_completed_only'] for e in episodes]), completion=mean([e['completion'] for e in episodes]),
                rpt=mean([e['restricted_person_time_per_generated'] for e in episodes])))
    group_rows = []; pairs = []; interaction = []
    for env, k, control in sorted({(t['environment'],t['k'],t['control']) for t in tasks}):
        reference = [r for r in rule_rows if (r['environment'],r['k'],r['control']) == (env,k,control)]
        # 规则先在验证锁定，不在测试上逐trace取最小。完成率门槛防止仅丢人而显得快。
        lock_path = Path(root) / 'rule_selection' / f'{env}__K{k}__{control}.json'
        if len(reference) == 3 and not lock_path.exists():
            highest = max(r['completion'] for r in reference)
            candidates = [r for r in reference if r['completion'] >= highest - .005 and r['co'] is not None]
            if candidates:
                save_json(lock_path, dict(selection=min(candidates,key=lambda r:(r['co'],r['rule'])),
                    scope='在验证数据上锁定；非数学下界，测试不重选',time=now()))
        locked = read_json(lock_path, {}).get('selection')
        for arm in ARMS:
            items = [r for r in results if (r['environment'],r['k'],r['control'],r['arm']) == (env,k,control,arm)]
            if not items: continue
            row = dict(environment=env,k=k,control=control,arm=arm,n=len(items),
                best_att=mean([r['best_co'] for r in items]),best_sd=sd([r['best_co'] for r in items]),
                plateau=mean([r['late_co'] for r in items]),within_run_sd=mean([r['late_sd'] for r in items]),
                completion=mean([r['best_completion'] for r in items]))
            if locked: row.update(rule=locked['rule'],rule_att=locked['co'],rule_completion=locked['completion'],
                delta_rule=row['best_att']-locked['co'],completion_gap=row['completion']-locked['completion'],
                co_eligible_vs_rule=row['completion']>=locked['completion']-.005)
            group_rows.append(row)
        for seed in sorted({r['seed'] for r in results}):
            selected = {r['arm']:r for r in results if (r['environment'],r['k'],r['control'],r['seed']) == (env,k,control,seed)}
            for arm, r in selected.items():
                for base in ('R_SOURCE','R00_NOALIGN','R10_S'):
                    if base in selected:
                        b = selected[base]
                        pairs.append(dict(environment=env,k=k,control=control,seed=seed,stage=r['stage'],arm=arm,reference=base,
                            delta_best=r['best_co']-b['best_co'],delta_plateau=r['late_co']-b['late_co'],
                            delta_completion=r['best_completion']-b['best_completion'],delta_rpt=r['best_rpt']-b['best_rpt'],
                            co_comparison_eligible=r['best_completion']>=b['best_completion']-.005))
            names=('R00_NOALIGN','R10_S','R01_AC','R11_SAC')
            if all(a in selected for a in names):
                a,b,c,d=(selected[n] for n in names)
                interaction.append(dict(environment=env,k=k,control=control,seed=seed,stage=a['stage'],
                    interaction_best=d['best_co']-b['best_co']-c['best_co']+a['best_co'],
                    interaction_plateau=d['late_co']-b['late_co']-c['late_co']+a['late_co'],
                    completion_range=max(r['best_completion'] for r in (a,b,c,d))-min(r['best_completion'] for r in (a,b,c,d)),
                    co_interaction_comparable=max(r['best_completion'] for r in (a,b,c,d))-min(r['best_completion'] for r in (a,b,c,d))<=.005,
                    interaction_rpt=d['best_rpt']-b['best_rpt']-c['best_rpt']+a['best_rpt']))
    save_csv(Path(root)/'RUN_STATUS.csv',status);save_csv(Path(root)/'METHOD_SEED_RESULTS.csv',results)
    save_csv(Path(root)/'GROUP_RESULTS.csv',group_rows);save_csv(Path(root)/'PAIRED_RESULTS.csv',pairs)
    save_csv(Path(root)/'INTERACTION_RESULTS.csv',interaction);save_csv(Path(root)/'RULE_RESULTS.csv',rule_rows)
    stage_rows=[]
    for stage,env,k,control,arm in sorted({(r['stage'],r['environment'],r['k'],r['control'],r['arm']) for r in results}):
        selected=[r for r in results if (r['stage'],r['environment'],r['k'],r['control'],r['arm'])==(stage,env,k,control,arm)]
        stage_rows.append(dict(stage=stage,environment=env,k=k,control=control,arm=arm,n=len(selected),
            best_att=mean([r['best_co'] for r in selected]),best_sd=sd([r['best_co'] for r in selected]),
            plateau=mean([r['late_co'] for r in selected]),completion=mean([r['best_completion'] for r in selected])))
    save_csv(Path(root)/'GROUP_RESULTS_BY_STAGE.csv',stage_rows)
    measured, uncertain = effect_budget(root)
    progress=dict(total=198,trained=sum(r['trained'] for r in status),evaluated=sum(r['evaluated'] for r in status),
        failed=sum(r['failed'] for r in status),completed_training_transitions=sum(r['steps'] for r in status if r['trained']),
        measured_training_work=measured,interrupted_unlogged_upper_bound=uncertain,
        planned_actual_budget=198_696_960,budget_cap=EFFECT_CAP,time=now())
    save_json(Path(root)/'progress.json',progress);return progress


def _bank_copy(paths, root, name, training_hash):
    """只复制用户明确给出的需求文件；不偷偷生成或修改分布。"""
    result=[];seen=set()
    for i, raw_path in enumerate(paths or []):
        path=Path(raw_path).resolve()
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if not {'id','time','origin_x','origin_y','dest_x','dest_y'} <= set(reader.fieldnames or []):
                raise RuntimeError('需求文件列不符合源接口')
        content=path.read_bytes();sha=hashlib.sha256(content).hexdigest()
        if sha==training_hash: raise RuntimeError('不能把训练trace冒充独立验证或测试trace')
        if sha in seen:raise RuntimeError('需求bank出现重复文件，不能伪增回放数')
        seen.add(sha)
        target=Path(root)/'banks'/name/f'{i:02d}_{sha[:12]}.csv'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=content: raise RuntimeError('需求bank已有异构内容')
        target.write_bytes(content);result.append(str(target))
    return result


def effect_main():
    parser=argparse.ArgumentParser(description='E0/严格E1同信息S×AC，198run，实际198.69696M；单文件可续训')
    parser.add_argument('--root');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--allow-code-change',action='store_true')
    parser.add_argument('--phase',choices=('all','train','eval','rules','pack','test'),default='all')
    parser.add_argument('--device',default='cuda');parser.add_argument('--eval-device',default='cpu')
    parser.add_argument('--timeout-hours',type=float,default=3.)
    parser.add_argument('--validation-traces',nargs='+');parser.add_argument('--test-traces',nargs='+')
    parser.add_argument('--worker',choices=('train','eval','rules','test'));parser.add_argument('--task')
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    if args.worker:
        record=read_json(args.task); t,c=record['task'],record['config'];root=Path(args.root).resolve()
        try:
            {'train':effect_train,'eval':effect_eval,'rules':effect_rules,
             'test':lambda t,r,c:effect_eval(t,r,c,True)}[args.worker](t,root,c)
        except KeyboardInterrupt: raise SystemExit(130)
        except Exception: traceback.print_exc();raise SystemExit(1)
        return
    tasks=effect_registry()
    if args.list:
        print(json.dumps(dict(version=EFFECT_VERSION,total=198,actual_budget=198_696_960,
            ppo=SOURCE_PPO,checkpoints=CHECKPOINT_STEPS,tasks=tasks),ensure_ascii=False,indent=2));return
    root=Path(args.root).resolve() if args.root else SCRIPT.parent/'serial_runs'/EFFECT_ROOT
    if args.phase=='pack':
        if not (root/'SUITE_PLAN.json').exists():raise RuntimeError('不是本轮结果目录')
        full_pack(root);return
    if args.timeout_hours<=0:parser.error('timeout-hours必须为正')
    if args.phase in ('all','train') and args.device.startswith('cuda') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('请先由用户查看top/nvidia-smi并显式指定CUDA_VISIBLE_DEVICES')
    if root.exists() and any(root.iterdir()) and not args.resume:raise RuntimeError('已有目录必须--resume，不覆盖旧结果')
    root.mkdir(parents=True,exist_ok=True);lock=strict_lock(root)
    try:
        native=materialize_source(root)
        old=read_json(root/'SUITE_PLAN.json')
        train_trace=str(native/'train_data'/'passengers_300.csv');train_hash=digest(train_trace)
        validations=_bank_copy(args.validation_traces,root,'validation',train_hash) if args.validation_traces else (old['config'].get('validation_traces',[]) if old else [])
        tests=_bank_copy(args.test_traces,root,'test',train_hash) if args.test_traces else []
        hashes=[digest(p) for p in validations]
        if tests and set(digest(p) for p in tests)&set(hashes):raise RuntimeError('测试与验证需求重复')
        config=dict(version=EFFECT_VERSION,device=args.device,eval_device=args.eval_device,trace=train_trace,
            train_horizon=500,eval_horizon=600,actual_steps=EFFECT_STEPS,requested_steps=1_000_000,
            resume=args.resume,retry_failed=args.retry_failed,timeout_hours=args.timeout_hours,
            validation_traces=validations,test_traces=tests,eval_seeds=[123,124],
            validation_scope='用户提供非同哈希需求验证；文件独立性不等于统计独立性已认证' if validations else '固定训练trace回放；不声称独立需求验证/泛化')
        contract=dict(version=EFFECT_VERSION,source=SOURCE_PAYLOAD_SHA256,sealed_runtime=_RUNTIME_SHA,
            source_ppo=SOURCE_PPO,fleet=12,train_trace=train_hash,validation_hashes=hashes,
            eval_seeds=[123,124],train_horizon=500,eval_horizon=600,tasks=tasks,checkpoints=CHECKPOINT_STEPS,
            schemas=[EffectSchema(k).manifest() for k in (2,3,4)],auxiliary=False,reward='原source奖励不改')
        signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        if old and old['signature']!=signature:raise RuntimeError('合同不同，拒绝混入同一目录')
        if old and old['script_sha256']!=digest(SCRIPT) and not args.allow_code_change:raise RuntimeError('实现代码改变；修复后续跑需显式--allow-code-change')
        save_json(root/'SUITE_PLAN.json',dict(signature=signature,contract=contract,config=config,
            script_sha256=digest(SCRIPT),time=now()))
        save_json(root/'TASK_REGISTRY.json',tasks);effect_aggregate(root,tasks)
        print('[预算] 198run；实际198,696,960；不改物理/PPO；'+config['validation_scope'],flush=True)
        # 原launch函数的test标记兼容由本轮包装，不调用旧矩阵调度器。
        if args.phase in ('all','rules'):
            for e,k,c in dict.fromkeys((t['environment'],t['k'],t['control']) for t in tasks):
                t=dict(id=f'RULES__{e}__K{k}__{c}',environment=e,k=k,control=c,fleet=12 if e=='E1' else None,
                    method=('J3__' if c=='J3' else 'SINGLE__')+'R_SOURCE',arm='R_SOURCE',seed=123)
                full_launch('rules',t,root,config);effect_aggregate(root,tasks)
        if args.phase in ('all','train'):
            for offset in range(0,len(tasks),6):
                block=tasks[offset:offset+6]
                for t in block:
                    print(f'[训练 {t["run_order"]}/198] {t["id"]}',flush=True)
                    full_launch('train',t,root,config);effect_aggregate(root,tasks)
                if args.phase=='all':
                    for t in block:
                        if (root/'cells'/t['id']/'TRAIN_END.json').exists():
                            full_launch('eval',t,root,config);effect_aggregate(root,tasks)
        if args.phase=='eval':
            for t in tasks:
                if (root/'cells'/t['id']/'TRAIN_END.json').exists():
                    full_launch('eval',t,root,config);effect_aggregate(root,tasks)
        if args.phase=='test':
            if not tests:raise RuntimeError('请显式给--test-traces；不会编造未见测试')
            selections={t['id']:read_json(root/'cells'/t['id']/'BEST_SELECTION.json') for t in tasks}
            if not all(selections.values()):raise RuntimeError('全部Best尚未锁定，不提前打开最终测试')
            test_contract=dict(test_hashes=[digest(p) for p in tests],selections=selections)
            prior=read_json(root/'TEST_PROTOCOL_LOCK.json')
            if prior and prior!=test_contract:raise RuntimeError('测试协议锁已存在且不同')
            save_json(root/'TEST_PROTOCOL_LOCK.json',test_contract)
            for t in tasks:effect_test_launch(t,root,config)
        progress=effect_aggregate(root,tasks)
        save_json(root/'SESSION_END.json',dict(phase=args.phase,progress=progress,time=now(),
            status='COMPLETE' if progress['evaluated']==198 and not progress['failed'] else 'PARTIAL_OR_FAILED'))
        save_json(root/'RUNNING.json',dict(phase='complete' if progress['evaluated']==198 else 'idle_partial',pid=None,task=None,time=now()))
        full_pack(root);print('[结束]',progress,flush=True)
    except KeyboardInterrupt:
        effect_aggregate(root,tasks);save_json(root/'SESSION_INTERRUPTED.json',dict(time=now(),note='相同命令--resume恢复'))
        raise SystemExit(130)
    finally:
        if read_json(lock,{}).get('pid')==os.getpid():lock.unlink(missing_ok=True)


def effect_test_launch(task,root,config):
    """复用隔离worker生命周期，但测试标记不冒充原验证EVAL_END。"""
    directory=Path(root)/'cells'/task['id']
    if (directory/'TEST_END.json').exists():return
    # 只生成自己的任务文件，子进程不会执行其他矩阵入口。
    save_json(directory/'TEST_TASK.json',dict(task=task,config=config))
    command=[sys.executable,'-u',str(SCRIPT),'--worker','test','--task',str(directory/'TEST_TASK.json'),'--root',str(root)]
    failed=directory/'FAILED_TEST.json'
    if failed.exists() and not config['retry_failed']:return
    process=None;started=time.time()
    try:
        with (directory/'test.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            while process.poll() is None:
                save_json(Path(root)/'RUNNING.json',dict(task=task,phase='test',pid=process.pid,elapsed_seconds=time.time()-started,time=now()))
                if time.time()-started>config['timeout_hours']*3600:raise TimeoutError('测试worker超时')
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        if process.returncode or not (directory/'TEST_END.json').exists():
            save_json(failed,dict(exit_code=process.returncode,time=now()))
        elif failed.exists():failed.rename(failed.with_name(f'FAILED_TEST_RESOLVED_{time.time_ns()}.json'))
    except (KeyboardInterrupt,Exception) as error:
        if process is not None and process.poll() is None:
            if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
        save_json(failed,dict(error=type(error).__name__,time=now()))
        if isinstance(error,KeyboardInterrupt):raise


if __name__ == '__main__':
    effect_main()
