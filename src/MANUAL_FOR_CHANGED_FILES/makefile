description_of_changed_files.pdf: description_of_changed_files.tex paper.bib

	pdflatex description_of_changed_files.tex

	bibtex description_of_changed_files

	pdflatex description_of_changed_files.tex

	pdflatex description_of_changed_files.tex

clean:
	rm description_of_changed_files.pdf description_of_changed_files.aux  description_of_changed_files.bbl  description_of_changed_files.blg  description_of_changed_files.log  description_of_changed_files.out  description_of_changed_files.toc

