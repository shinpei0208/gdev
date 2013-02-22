#!/bin/bash



# nvclock-lookup.sh vbios.rom

nvbios $1 > ${1}.dump

if [ $? -eq 1 -o $# -eq 0 ] 
then
	echo -e "Please input vbios data!\nLike this: ./nvclock-lookup.sh [filename]"
	rm -f ${1}.dump
	exit 0
fi

echo "$1"


SetCount=`grep Core ${1}.dump | wc -l` 

Memory=( 0 )
Core=( 0 )
i=1

while [ ${SetCount} -ge ${i} ]
do 
	Memory[${i}]=`grep Core ${1}.dump | sed -n ${i}p | cut -f7 -d " "`	
	Core[${i}]=`grep Core ${1}.dump | sed -n ${i}p | cut -f5 -d " "`	
	i=`expr ${i} + 1 `
done	


# ${${Conpornent}[${i}]}
	
i=1
j=0
echo -n -e "You can select Memory frequency\t:\t"
while [ ${SetCount} -ge ${i} ]
do 
	if [ ${Memory[${i}]} != ${Memory[${j}]} ]
	then
		echo -n -e "${Memory[${i}]}\t"
	

	fi

	i=`expr ${i} + 1 `
	j=`expr ${j} + 1 `
	
done
echo 

i=1
j=0
echo -n -e "You can select Core frequency\t:\t"
while [ ${SetCount} -ge ${i} ]
do 
	if [ ${Core[${i}]} != ${Core[${j}]} ]
	then
		echo -n -e "${Core[${i}]}\t"
	

	fi

	i=`expr ${i} + 1 `
	j=`expr ${j} + 1 `

done
echo



