
all: logfiles/lecture/*

logfiles/grokking/%.txt: grokking/%.py
	echo import grokking.$% | venv/bin/python3 > $@;

logfiles/lecture/%.txt: lecture/%.py
	echo import lecture.$* | venv/bin/python3 > $@;
