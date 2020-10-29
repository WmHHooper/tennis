
all: logfiles/*/*

logfiles/grokking/%.txt: netExamples/grokking/%.py
	touch $@
	# echo import grokking.$% | venv/bin/python3 > $@;

logfiles/lecture/%.txt: netExamples/lecture/%.py
	echo import $< | sed s/\.py// | sed s,/,.,g | venv/bin/python3 | tee $@;

logfiles/election/%.txt: netExamples/election/%.py
	echo import $< | sed s/\.py// | sed s,/,.,g | venv/bin/python3 | tee $@;

logfiles/solution/%.txt: netExamples/solution/%.py
	echo import $< | sed s/\.py// | sed s,/,.,g | venv/bin/python3 | tee $@;

targets:
	for p in netExamples/*/*.py
	do
	b=`basename $p | sed s/\.py//`
	e=`dirname $p`
	d=`basename $e`
	echo make logfiles/$d/$b.txt
	done
