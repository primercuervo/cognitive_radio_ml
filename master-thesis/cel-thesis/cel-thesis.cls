% Author: Johannes Demel
% This file requires quite a few texlive packages.
% texlive-lang-german / texlive-lang-european
% texlive-science

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cel-thesis/cel-thesis}
\LoadClass[11pt,	% 11pt
	    titlepage,
	    BCOR=8mm, 		% Bindekorrektur
	    DIV=12,		% Satzspiegel
	    twoside,		% Doppelseitige Ausgabe
	    a4paper,		% DIN A4
	    openright,		% Neue Kapitel auf rechter Seite
%	    dvips,		% Source Specials ins DVI/PS
%	    draft,		% Overfull H-Boxen markieren
	    final,		% Engueltige Version mit Bildern
]{scrreprt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new commands
\newcommand{\thesisTitle}[1]{\def \theThesisTitle{#1}}
\newcommand{\thesisType}[1]{\def \theThesisType{#1}}
\newcommand{\thesisAuthor}[1]{\def \theThesisAuthor{#1}}
\newcommand{\thesisAdvisor}[1]{\def \theThesisAdvisor{#1}}
\newcommand{\thesisSupervisor}[1]{\def \theThesisSupervisor{#1}}
\newcommand{\thesisHeadOfInstitute}[1]{\def \theThesisHeadOfInstitute{#1}}
\newcommand{\thesisStartDate}[1]{\def \theThesisStartDate{#1}}
\newcommand{\thesisEndDate}[1]{\def \theThesisEndDate{#1}}
\newcommand{\thesisSignatureDate}[1]{\def \theThesisSignatureDate{#1}}
\newcommand{\thesisLanguage}[1]{\def \theThesisLanguage{#1}}
\newcommand{\thesisCC}[1]{\def \theThesisCC{#1}}
\newcommand{\thesisPythonWatermark}[1]{\def \theThesisPythonWatermark{#1}}
% Defaults
\thesisTitle{Random Thesis Title}
\thesisType{Master Thesis Type}
\thesisAuthor{Nicolas Cuervo Benavides}
\thesisAdvisor{Dr.-Ing. Holger J\"akel}
\thesisHeadOfInstitute{Univ.-Prof. i.R. Dr.rer.nat. Friedrich K. Jondral}
\thesisSupervisor{Super-Assi Deluxe M.Sc.}
\thesisStartDate{dd.mm.yyyy}
\thesisEndDate{dd.mm.yyyy}
\thesisSignatureDate{dd.mm.yyyy}
\thesisLanguage{english}
\thesisCC{TRUE}
\thesisPythonWatermark{FALSE}

\def\dictThesisLanguageGerman{ngerman}
\def\dictThesisLanguageEnglish{english}
\def\dictThesisFALSE{FALSE}
\def\dictThesisTRUE{TRUE}


% Gerneral %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage[T1]{fontenc}  % encoding for target files.
\usepackage[utf8]{inputenc} % UTF-8 encoding for source files
\usepackage[printonlyused]{acronym} % use this package for abbreviations!
\def\setlanguage{
    \usepackage[\theThesisLanguage]{babel}
    \ifx\theThesisLanguage\dictThesisLanguageEnglish
        \bibliographystyle{alpha} % American Style
        \def\theUniversity{Karlsruhe Institute of Technology}
        \sisetup{%
            binary-units = true,
            output-decimal-marker={.}
        }
    \else
        \bibliographystyle{alphadin} % DIN norm
        \def\theUniversity{Karlsruher Institut für Technologie}
        \sisetup{%
            binary-units = true,
        output-decimal-marker={,}
        }
        \mathcode`,="013B % avoid space in comma decimal numbers
    \fi
}

% Print Watermark! CAREFUL: if does not work %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 \ifx\theThesisPythonWatermark\dictThesisTRUE
 \usepackage{draftwatermark}
 \SetWatermarkText{\includegraphics{cel-thesis/logos/watermark_thesis.pdf}}
 \SetWatermarkScale{1.0}
 \SetWatermarkAngle{0}
 \fi

% Math and Symbols %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{amssymb, amsfonts, latexsym}
\usepackage{amsmath}
\usepackage[right]{eurosym}
\usepackage{siunitx}

%% Define always missing math symbols.
\DeclareMathOperator*{\argmin}{\arg\!\min}
\DeclareMathOperator*{\argmax}{\arg\!\max}
\DeclareMathOperator\sign{sign}



% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ueberschriften fett
\setkomafont{sectioning}{\normalcolor\bfseries}
% Bild- und Tabellen-Unterschriften formatieren
\renewcommand{\caplabelfont}{\bfseries}
\renewcommand{\capfont}{\small}

\usepackage{graphicx}
\usepackage{color}
\usepackage{pgf, tikz, pgfplots}
\usetikzlibrary{shapes,arrows}
\pgfplotsset{compat=1.9}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{marginnote}
\RequirePackage{ifthen}
\RequirePackage[hidelinks, final=true]{hyperref}
\RequirePackage[ddmmyyyy]{datetime}
\renewcommand{\dateseparator}{.}
\RequirePackage[export]{adjustbox}
\setlength{\parindent}{0pt}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create titlepage and optional parts like a CC statement.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\header}{%
    \includegraphics[width=40mm,valign=c]{cel-thesis/logos/kitlogo_de_rgb}%
    \hfill%
    \begin{minipage}{8.26cm}\centering%
        \textbf{\theUniversity}\\
        Communications Engineering Lab\\
        \theThesisHeadOfInstitute%
    \end{minipage}%
    \hfill%
    \includegraphics[width=25mm,valign=c]{cel-thesis/logos/cel_logo}%
}
\newcommand{\ssection}[1]{\vspace*{0.8em}{\textbf{#1}}\\}

\def\makestatutorydeclaration{
  \include{cel-thesis/eidesstattliche_erklaerung}
}

\def\maketitle{
    \hypersetup{
        pdfauthor = \theThesisAuthor,
        pdftitle = {\theThesisType{ }\theThesisTitle{ } \theThesisAuthor},
        pdfsubject = {Institut für Nachrichtentechnik (CEL)},
        pdfcreator = {PDFLaTeX with hyperref package},
        pdfproducer = {PDFLaTeX}
    }
    \thispagestyle{empty}
    \header  % institute banner, etc.
    \vspace*{5.0cm}%
    % Title
    \begin{center}
        {\Large \textbf{\theThesisTitle}} \\[1cm]
        {\large \theThesisType} \\[2.5cm]
        {\textbf{\theThesisAuthor}}\\[5cm]
        \begin{tabular}{lll}
        \ifx\theThesisLanguage\dictThesisLanguageEnglish  % default is german. But one may use english.
	  Advisor 	& : & \theThesisAdvisor \\
	  Supervisor 	& : & \theThesisSupervisor\\[2cm]
	  Start date 	& : & \theThesisStartDate \\
	  End date  	& : & \theThesisEndDate \\
	\else
	  Hauptreferent & : & \theThesisAdvisor \\
	  Betreuer      & : & \theThesisSupervisor\\[2cm]
	  Beginn        & : & \theThesisStartDate \\
	  Abgabe        & : & \theThesisEndDate \\
	\fi
	\end{tabular}

    \end{center}
    \hspace*{\fill}
    \newpage
    \makestatutorydeclaration
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data visualization styles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%grid style
\pgfplotsset{
  grid style = {
    dash pattern = on 0.05mm off 1mm,
    line cap = round,
    black,
    opacity=0.5,
    line width = 0.2pt
  }
}
