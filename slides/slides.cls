\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{slides}

\newif\if@loadpgfpages

\DeclareOption{handout}{\@loadpgfpagestrue \PassOptionsToClass{\CurrentOption}{beamer}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax

% class
\LoadClass{beamer}
\usepackage{bookmark}

% handout
\if@loadpgfpages
    \usepackage{pgfpages}
    \pgfpagesuselayout{6 on 1}[a4paper,border shrink=3mm]
    \pgfpageslogicalpageoptions{1}{border code=\pgfusepath{stroke}, center = \pgfpoint{.275\pgfphysicalwidth}{.80\pgfphysicalheight}}
    \pgfpageslogicalpageoptions{2}{border code=\pgfusepath{stroke}, center = \pgfpoint{.725\pgfphysicalwidth}{.80\pgfphysicalheight}}
    \pgfpageslogicalpageoptions{3}{border code=\pgfusepath{stroke}, center = \pgfpoint{.275\pgfphysicalwidth}{.50\pgfphysicalheight}}
    \pgfpageslogicalpageoptions{4}{border code=\pgfusepath{stroke}, center = \pgfpoint{.725\pgfphysicalwidth}{.50\pgfphysicalheight}}
    \pgfpageslogicalpageoptions{5}{border code=\pgfusepath{stroke}, center = \pgfpoint{.275\pgfphysicalwidth}{.20\pgfphysicalheight}}
    \pgfpageslogicalpageoptions{6}{border code=\pgfusepath{stroke}, center = \pgfpoint{.725\pgfphysicalwidth}{.20\pgfphysicalheight}}
\fi

% toc display section frame number
\makeatletter
\long\def\beamer@section[#1]#2{%
  \beamer@savemode%
  \mode<all>%
  \ifbeamer@inlecture
    \refstepcounter{section}%
    \beamer@ifempty{#2}%
    {\long\def\secname{#1}\long\def\lastsection{#1}}%
    {\global\advance\beamer@tocsectionnumber by 1\relax%
      \long\def\secname{#2}%
      \long\def\lastsection{#1}%
      \addtocontents{toc}{\protect\beamer@sectionintoc{\the\c@section}{#2\hfill\the\c@page}{\the\c@page}{\the\c@part}%
        {\the\beamer@tocsectionnumber}}}%
    {\let\\=\relax\xdef\sectionlink{{Navigation\the\c@page}{\noexpand\secname}}}%
    \beamer@tempcount=\c@page\advance\beamer@tempcount by -1%
    \beamer@ifempty{#1}{}{%
      \addtocontents{nav}{\protect\headcommand{\protect\sectionentry{\the\c@section}{#1}{\the\c@page}{\secname}{\the\c@part}}}%
      \addtocontents{nav}{\protect\headcommand{\protect\beamer@sectionpages{\the\beamer@sectionstartpage}{\the\beamer@tempcount}}}%
      \addtocontents{nav}{\protect\headcommand{\protect\beamer@subsectionpages{\the\beamer@subsectionstartpage}{\the\beamer@tempcount}}}%
    }%
    \beamer@sectionstartpage=\c@page%
    \beamer@subsectionstartpage=\c@page%
    \def\insertsection{\expandafter\hyperlink\sectionlink}%
    \def\insertsubsection{}%
    \def\insertsubsubsection{}%
    \def\insertsectionhead{\hyperlink{Navigation\the\c@page}{#1}}%
    \def\insertsubsectionhead{}%
    \def\insertsubsubsectionhead{}%
    \def\lastsubsection{}%
    \Hy@writebookmark{\the\c@section}{\secname}{Outline\the\c@part.\the\c@section}{2}{toc}%
    \hyper@anchorstart{Outline\the\c@part.\the\c@section}\hyper@anchorend%
    \beamer@ifempty{#2}{\beamer@atbeginsections}{\beamer@atbeginsection}%
  \fi%
  \beamer@resumemode}%

\def\beamer@subsection[#1]#2{%
  \beamer@savemode%
  \mode<all>%
  \ifbeamer@inlecture%
    \refstepcounter{subsection}%
    \beamer@ifempty{#2}{\long\def\subsecname{#1}\long\def\lastsubsection{#1}}
    {%
      \long\def\subsecname{#2}%
      \long\def\lastsubsection{#1}%
      \addtocontents{toc}{\protect\beamer@subsectionintoc{\the\c@section}{\the\c@subsection}{#2\hfill\the\c@page}{\the\c@page}{\the\c@part}{\the\beamer@tocsectionnumber}}%
    }%
    \beamer@tempcount=\c@page\advance\beamer@tempcount by -1%
    \addtocontents{nav}{%
      \protect\headcommand{\protect\beamer@subsectionentry{\the\c@part}{\the\c@section}{\the\c@subsection}{\the\c@page}{\lastsubsection}}%
      \protect\headcommand{\protect\beamer@subsectionpages{\the\beamer@subsectionstartpage}{\the\beamer@tempcount}}%
    }%
    \beamer@subsectionstartpage=\c@page%
    \edef\subsectionlink{{Navigation\the\c@page}{\noexpand\subsecname}}%
    \def\insertsubsection{\expandafter\hyperlink\subsectionlink}%
    \def\insertsubsubsection{}%
    \def\insertsubsectionhead{\hyperlink{Navigation\the\c@page}{#1}}%
    \def\insertsubsubsectionhead{}%
    \Hy@writebookmark{\the\c@subsection}{#2}{Outline\the\c@part.\the\c@section.\the\c@subsection.\the\c@page}{3}{toc}%
    \hyper@anchorstart{Outline\the\c@part.\the\c@section.\the\c@subsection.\the\c@page}\hyper@anchorend%
    \beamer@ifempty{#2}{\beamer@atbeginsubsections}{\beamer@atbeginsubsection}%
  \fi%
  \beamer@resumemode}
\makeatother

% global customizations
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\setbeamertemplate{itemize item}[triangle]
\setbeamertemplate{itemize subitem}[circle]
\setbeamertemplate{itemize subsubitem}[square]
\setbeamerfont{footnote}{size=\tiny}

% font
\usepackage{txfonts}
\linespread{1.4}\selectfont

% figures
\usepackage{graphicx}
\usepackage{subcaption}
\captionsetup{justification=centering,singlelinecheck=false}

% box
\usepackage{tcolorbox}
\tcbset{fontupper=\small, fonttitle=\normalsize, boxsep=0.15mm, left=1mm, right=1mm, top=1mm, bottom=1mm}

% links
\usepackage{ulem}
\usepackage{url}
\urlstyle{same}
\newcommand\link[1]{\dotuline{\url{#1}}}
\newcommand\hlink[2]{\dotuline{\href{#1}{#2}}}

% maths
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{mathtools}
\DeclareMathOperator{\E}{E}
\DeclareMathOperator{\adj}{adj}
\DeclarePairedDelimiter\abs{\lvert}{\rvert}
