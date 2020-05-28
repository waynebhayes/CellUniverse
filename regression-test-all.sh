#!/bin/sh
RETURN=0
#renice 19 $$
STDBUF=''
if which stdbuf >/dev/null; then
	STDBUF='stdbuf -oL -eL'
fi
for dir in regression-tests/*; do
	REG_DIR=$dir
	export REG_DIR
	echo --- in directory $dir ---
	for r in $dir/*.sh; do
		echo --- running test $r ---
		if eval $STDBUF "$r"; then # force output and error to be line buffered
			:
		else
			RETURN=1
		fi
	done
done
exit $RETURN
