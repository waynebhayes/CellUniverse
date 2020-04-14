#!/bin/sh
RETURN=0
#renice 19 $$
for dir in regression-tests/*; do
	echo --- in directory $dir ---
	for r in $dir/*.sh; do
		echo --- running test $r ---
		if "$r"; then
			:
		else
			RETURN=1
		fi
	done
done
exit $RETURN
