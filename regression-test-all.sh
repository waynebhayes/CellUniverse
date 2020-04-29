#!/bin/sh
RETURN=0
#renice 19 $$
for dir in regression-tests/*; do
	echo --- in directory $dir ---
	for r in $dir/*.sh; do
		echo --- running test $r ---
		if stdbuf -oL -eL "$r"; then # force output and error to be line buffered
			:
		else
			RETURN=1
		fi
	done
done
exit $RETURN
