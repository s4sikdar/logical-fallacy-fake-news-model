#!/usr/bin/env bash

tempfile="$(mktemp)"

cd gossipcop_fake/
for filename in ./*.html
do
	filename_base=$(echo "${filename}" | awk 'BEGIN { FS = "." } ; { print $2 }' | awk 'BEGIN { FS = "/" } ; { print $2 }')
	text_file="${filename_base}.txt"
	if [ -f "${text_file}" ]
	then
		mv "${filename_base}.html" "${filename_base}-false.html"
		mv "${filename_base}.txt" "${filename_base}-false.txt"
	fi
done
cd ../


cd gossipcop_real/
for filename in ./*.html
do
	filename_base=$(echo "${filename}" | awk 'BEGIN { FS = "." } ; { print $2 }' | awk 'BEGIN { FS = "/" } ; { print $2 }')
	text_file="${filename_base}.txt"
	if [ -f "${text_file}" ]
	then
		mv "${filename_base}.html" "${filename_base}-true.html"
		mv "${filename_base}.txt" "${filename_base}-true.txt"
	fi
done
cd ../


cd politifact_fake/
for filename in ./*.html
do
	filename_base=$(echo "${filename}" | awk 'BEGIN { FS = "." } ; { print $2 }' | awk 'BEGIN { FS = "/" } ; { print $2 }')
	text_file="${filename_base}.txt"
	if [ -f "${text_file}" ]
	then
		mv "${filename_base}.html" "${filename_base}-false.html"
		mv "${filename_base}.txt" "${filename_base}-false.txt"
	fi
done
cd ../


cd politifact_real/
for filename in ./*.html
do
	filename_base=$(echo "${filename}" | awk 'BEGIN { FS = "." } ; { print $2 }' | awk 'BEGIN { FS = "/" } ; { print $2 }')
	text_file="${filename_base}.txt"
	if [ -f "${text_file}" ]
	then
		mv "${filename_base}.html" "${filename_base}-true.html"
		mv "${filename_base}.txt" "${filename_base}-true.txt"
	fi
done
cd ../

rm ${tempfile}
