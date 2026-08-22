%% ---------------------------------------------------------------------------
%% svjour3.cls -- LOCAL PREVIEW SHIM, NOT the Springer class.
%%
%% Springer distributes the real svjour3 class under a licence that keeps it off
%% CTAN, and this build environment has no network route to Springer's servers.
%% This shim (copied from another project's local preview tool, itself built for
%% the same reason) exists only so this manuscript can be compiled locally for
%% proofreading before actual submission to Computational Economics.
%%
%% It implements the subset of the svjour3 interface this manuscript uses:
%%   \journalname, \smartqed, \institute, \at, \email, \keywords, \titlerunning,
%%   \authorrunning, and the acknowledgements environment. Single-column layout
%% (most Computational Economics articles are single-column, and this
%% manuscript's wide tabularx tables need the full text width).
%%
%% Page geometry and typography are approximations of the Springer layout, so
%% page counts here are indicative only. Replace this file with Springer's
%% official svjour3.cls (from the journal's LaTeX author template) before
%% actual submission -- the source in main.tex targets the genuine class and
%% should not need to change when the real class file is dropped in.
%% ---------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{svjour3}[2026/08/22 Local preview shim for Springer svjour3]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,a4paper]{article}

\RequirePackage[margin=2.5cm]{geometry}
\RequirePackage{times}

% --- svjour3 interface bits -------------------------------------------------
\newcommand{\journalname}[1]{\gdef\@journalname{#1}}
\gdef\@journalname{}
\newcommand{\smartqed}{}
\newcommand{\at}{\\}
\newcommand{\titlerunning}[1]{}
\newcommand{\authorrunning}[1]{}

% article defines neither \institute nor \@instituteblock, so both are new here.
\newcommand{\@instituteblock}{}
\newcommand{\institute}[1]{\gdef\@instituteblock{#1}}

\providecommand{\email}[1]{\texttt{#1}}

% svjour3 uses \and as a keyword separator.  article defines \and for the
% \author alignment, where it expands to \end{tabular}...\begin{tabular}, so it
% must be rebound locally or the keyword list detonates the surrounding group.
\newcommand{\keywords}[1]{%
  \par\smallskip\noindent
  \begingroup
    \renewcommand{\and}{\;\textperiodcentered\;}%
    \textbf{Keywords}\quad #1\par
  \endgroup}

\newenvironment{acknowledgements}%
  {\par\medskip\noindent\textbf{Acknowledgements}\quad}%
  {\par}

% Springer numbers theorems within the document, not per section.
\renewcommand{\thesection}{\arabic{section}}

% --- title block ------------------------------------------------------------
% Single-column, unlike the fc-gnn shim this was copied from: Computational
% Economics articles are single-column, and this manuscript's wide tabularx
% tables need the full text width, not a half-page column.
\renewcommand{\maketitle}{%
  \ifx\@journalname\empty\else
    {\footnotesize\itshape\@journalname\par\medskip}
  \fi
  {\raggedright\LARGE\bfseries\@title\par}
  \medskip
  {\raggedright\large\@author\par}
  \medskip
  {\raggedright\footnotesize\@instituteblock\par}
  \medskip
  {\footnotesize\@date\par}
  \bigskip
}

% Springer's abstract is a run-in block, not a centred starred section.
\renewenvironment{abstract}%
  {\small\noindent\textbf{Abstract}\quad}%
  {\par\medskip}

\endinput
