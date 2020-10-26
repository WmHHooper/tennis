
all: logfiles/*/*

logfiles/grokking/%.txt: netExamples/grokking/%.py
	touch $@
	# echo import grokking.$% | venv/bin/python3 > $@;

logfiles/lecture/%.txt: netExamples/lecture/%.py
	echo import $< | sed s/\.py// | sed s,/,.,g | venv/bin/python3 > $@;

logfiles/election/%.txt: netExamples/election/%.py
	echo import $< | sed s/\.py// | sed s,/,.,g | venv/bin/python3 > $@;

logfiles/solution/%.txt: netExamples/solution/%.py
	echo import $< | sed s/\.py// | sed s,/,.,g | venv/bin/python3 > $@;
