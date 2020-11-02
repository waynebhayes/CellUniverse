#!/bin/sh
if [ -f git-at -a `git log -1 --format=%at` -eq `tail -1 git-at` ]; then
    echo -n "Repo unchanged; returning same status code as "
    tail -1 git-at | xargs -I{} date -d @{} +%Y-%m-%d-%H:%M:%S
    exit `head -1 git-at`
fi

git submodule init && git submodule update && (cd NetGO && git checkout master && git pull)
nice -19 ./regression-test-all.sh -make
(echo $?; git log -1 --format=%at) > git-at


NUM_FAILS=0
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
	    (( NUM_FAILS+=$? ))
	fi
    done
done
(echo $NUM_FAILS; git log -1 --format=%at) > git-at
exit $NUM_FAILS
