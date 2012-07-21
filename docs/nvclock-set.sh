#!/bin/bash
# Please reboot, you want to clock level up or you want to use help (Option -h)
# nvclock-set.sh -c 700 -m 1848 vbios.rom

nvbios $5 > ${5}.dump
modprobe -r nvidia

if [ $? -eq 1 -o $# -ne 5 ] 
then
	echo -e "Please input vbios data and set frequency!\nLike this: ./nvclock-set.sh -c [Core frequency] -m Memory frequency] [filename]"
	rm -f ${5}.dump
	exit 0
fi



SetCount=`grep Core ${5}.dump | wc -l` 

NumResister=0
SetLevel=0

# check frequency
Memory=( 0 )
Core=( 0 )
i=1

while [ ${SetCount} -ge ${i} ]
do 
	Memory[${i}]=`grep Core ${5}.dump | sed -n ${i}p | cut -f7 -d " "|cut -f1 -d "M"`	
	Core[${i}]=`grep Core ${5}.dump | sed -n ${i}p | cut -f5 -d " "|cut -f1 -d "M"`	
	i=`expr ${i} + 1 `
done	


if [ $1 = -c -o $3 = -c ]
then	
	if [ $1 = -c ]
	then 
		FreqCore=$2
	else
		FreqCore=$4
	fi
	
fi


if [ $1 = -m -o $3 = -m ]
then	
	if [ $1 = -m ]
	then 
		FreqMem=$2
	else
		FreqMem=$4
	fi
	
fi

i=${SetCount}
while [ $i -gt 0 ]
do
	if [ ${FreqCore} = ${Core[${i}]} ]
	then
		break
	fi
	i=`expr $i - 1`
	if [ $i -eq 0 ]
	then
		echo "Please input valid core frequency!"
	fi
done

j=${SetCount}
while [ $j -gt 0 ]
do
	if [ ${FreqMem} -eq ${Memory[${j}]}  ]
	then
		break
	fi
	
	j=`expr $j - 1`
	if [ $j -eq 0 ]
	then
		echo "Please input valid memory frequency!"
		exit 0
	fi


done

# finsh check

# set core frequency 
while [ ${SetCount} -gt $i ]
do
	SetLevel=`expr ${SetCount} \* 3 - 1 `
	CoreNumResister=`grep -A 1 Core ${5}.dump | sed -n ${SetLevel}p |cut -f1 -d ":"`
	SetCount=`expr ${SetCount} - 1 `
	CoreOption=`echo "${CoreOption} -e ${CoreNumResister}:ff"`
done

CoreOption=`echo "${CoreOption} -e ${CoreNumResister}:0${FreqCore0x}"`

# set memory frequency
if [ ${FreqMem} -lt 1000 ]
then
	FreqMemBin=`echo "obase=2; ibase=10; ${FreqMem} + 32768 " |bc`
	FreqMemBin=`echo "obase=16; ibase=2; ${FreqMemBin} " |bc |tr A-Z a-z`
else
	FreqMemBin=`echo "obase=2; ibase=10; ${FreqMem} + 16384 " |bc`
	FreqMemBin=`echo "obase=16; ibase=2; ${FreqMemBin} " |bc |tr A-Z a-z`
fi

FreqMem0x1=`echo ${FreqMemBin} |cut -c3-4`
FreqMem0x2=`echo ${FreqMemBin} |cut -c1-2`
	
SetLevel=`expr ${SetCount} \* 9 - 1 `
MemNumResister=`grep -A 7 Core ${5}.dump | sed -n ${SetLevel}p |cut -f2 -d ":"`

MemOption=`echo  " -e ${MemNumResister}:${FreqMem0x1}"`
MemNumResister=`echo ${MemNumResister} |tr a-z A-Z`
MemNumResister=`echo "obase=16; ibase=16; ${MemNumResister} +1 " |bc |tr A-Z a-z`


MemOption=`echo -n "${MemOption} -e ${MemNumResister}:${FreqMem0x2}"`
echo "nvafakebios${CoreOption}${MemOption} ${5}"
nvafakebios ${CoreOption} ${MemOption} ${5}
modprobe nvidia
modprobe -r nvidia
nvagetbios > ${5}
nvbios ${5} > ${5}.dump

echo "GPU run this frequency"
grep Core ${5}.dump | sed -n ${i}p | cut -f4-9 -d " "
rm -f ${5}.dump
modprobe nvidia


