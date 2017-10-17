SET p=0
SET k=-1
SET m=2

SET s1=0
SET s2=9001
SET s3=35001
SET s4=67001
SET f1=9000
SET f2=35000
SET f3=67000
SET f4=-1

IF "%p%"==1 (
	SET s2=3001
	SET s3=8001
	SET s4=15001
	SET f1=3000
	SET f2=8000
	SET f3=15000
)

start python pgt.py -m %m% -p %p% -k %k% -s %s1% -f %f1%
start python pgt.py -m %m% -p %p% -k %k% -s %s2% -f %f2%
start python pgt.py -m %m% -p %p% -k %k% -s %s3% -f %f3%
start python pgt.py -m %m% -p %p% -k %k% -s %s4% -f %f4%

pause