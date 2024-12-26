\ProvidesClass{report}[report]

\LoadClass[11pt,letterpaper]{article}

\usepackage[left=1.4cm,top=1.5cm,right=1.4cm,bottom=1.5cm]{geometry}
\usepackage[parfill]{parskip}
\usepackage[dvipsnames]{xcolor}
\usepackage{lmodern}
\usepackage{fontspec}
\usepackage{kotex}

\setlength\parindent{6pt}

\renewcommand\maketitle{
  {
    \raggedright
    \begin{center}
    { \huge \bfseries \@title }\\[2ex]
    { \Large  \@author }\\[1ex] 
    \@date\\[4ex]
    \end{center}
  }
}

\newcommand\divider{\noindent\textcolor{lightgray}{\rule{\textwidth}{0.4pt}}}
\newcommand{\code}[1]{\texttt{#1}}
% https://tex.stackexchange.com/a/219497 for line wraping in \texttt
\catcode`_=12 %
\renewcommand{\texttt}[1]{%
  \begingroup
  \ttfamily
  \begingroup\lccode`~=`/\lowercase{\endgroup\def~}{/\discretionary{}{}{}}%
  \begingroup\lccode`~=`[\lowercase{\endgroup\def~}{[\discretionary{}{}{}}%
  \begingroup\lccode`~=`.\lowercase{\endgroup\def~}{.\discretionary{}{}{}}%
  \begingroup\lccode`~=`_\lowercase{\endgroup\def~}{_\discretionary{}{}{}}%
  \catcode`/=\active\catcode`[=\active\catcode`.=\active\catcode`_=\active
  \scantokens{#1\noexpand}%
  \endgroup
}
\catcode`_=8 % 

\setmainfont{Noto Sans}[UprightFont=* Light, BoldFont=* SemiBold]

\setmainhangulfont{Noto Sans CJK KR}[
  UprightFont=* Light, BoldFont=* Bold,
  Script=Hangul, Language=Korean, AutoFakeSlant,
]
\setsanshangulfont{Noto Sans CJK KR}[
  UprightFont=* DemiLight, BoldFont=* Medium,
  Script=Hangul, Language=Korean
]
\setmathhangulfont{Noto Sans CJK KR}[
  SizeFeatures={
    {Size=-6,  Font=* Medium},
    {Size=6-9, Font=*},
    {Size=9-,  Font=* DemiLight},
  },
  Script=Hangul, Language=Korean
]

\def\sectionskipbefore{\medskip}
\def\sectionskipafter{\medskip}
\def\sectionskip{\medskip}
