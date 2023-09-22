#!/bin/bash

# usage
if [ $# -ne 1 ] ; then
 echo ""
 echo 'Usage: bulk-whois.sh <input file>'
 echo 'Example: bulk-whois.sh infile'
 echo 'Output will be in infile.whois'
 echo ""
 exit
fi

# varable
INFILE=$1

# check for netcat
if [ -n "$NETCAT" ] ; then
 alias NETCAT="$NETCAT"
elif which netcat &> /dev/null ; then
 alias NETCAT='netcat'
else
 echo 'You need netcat to run this!'
fi

# format the input file
sed -e '1ibegin' -e '1icountrycode' $INFILE > $INFILE.f
echo 'end' >> $INFILE.f

# query cymru whois
`netcat whois.cymru.com 43 < $INFILE.f > $INFILE.whois`

# clean up files
rm $INFILE.f