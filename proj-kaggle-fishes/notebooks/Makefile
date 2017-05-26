# Filename
SRC = Article2.md
THEME := "Frankfurt"

PDFS=$(SRC:.md=.pdf)
HTML=$(SRC:.md=.html)

all:	$(HTML)

pdf:	clean $(PDFS)
html:	clean $(HTML)

%.html:	%.md
	pandoc -s -o $@ --css=pandoc.css $< &

%.pdf:	%.md
	pandoc -t beamer -s -V theme=$(THEME) --latex-engine=pdflatex -o $@ $< &

clean:
	rm -f *.html *.pdf
