#!/bin/bash
# Initialize
args=0
p=0
k=0
m=0

while [ "$#" -gt 0 ]; do
  args=1
  case "$1" in
    -h)
        echo "run_pgt.sh <args>"
        echo "-h for help"
        echo "-m MODE"
        echo "-p 0 for Gowalla and 1 for Brightkite"
        echo "-k top k users, or 0 for weekend, and -1 for all data"
        exit 0
        ;;
    -m)  m="$2";
        shift 2
        ;;
    -p)  p="$2";
        shift 2
        ;;
    -k)  k="$2";
        shift 2
        ;;
    -*) echo "unknown option: $1" >&2; exit 1;;
    *) handle_argument "$1"; shift 1;;
  esac
done

if [ $args != 0 ];
  then 
    # Gowalla
    s01=0
    s02=10001
    s03=30001
    s04=55001
    f01=10000
    f02=30000
    f03=55000
    f04=-1
    # Brightkite
    s11=0
    s12=3001
    s13=8001
    s14=15001
    s15=30001
    f11=3000
    f12=8000
    f13=15000
    f14=30000
    f15=-1
    if [ $p == 0 ];
    then
	# Gowalla
	python pgt.py -m $m -p $p -k $k -s $s01 -f $f01 &
	python pgt.py -m $m -p $p -k $k -s $s02 -f $f02 &
	python pgt.py -m $m -p $p -k $k -s $s03 -f $f03 &
	python pgt.py -m $m -p $p -k $k -s $s04 &
    elif [ $p == 1 ];
    then
	# Brightkite
	python pgt.py -m $m -p $p -k $k -s $s11 -f $f11 &
	python pgt.py -m $m -p $p -k $k -s $s12 -f $f12 &
	python pgt.py -m $m -p $p -k $k -s $s13 -f $f13 &
	python pgt.py -m $m -p $p -k $k -s $s14 -f $f14 &
	python pgt.py -m $m -p $p -k $k -s $s15 &
    fi
fi
