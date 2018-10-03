#!/usr/bin/python2

from pymongo import MongoClient
import sys
from pprint import pprint
import bz2

# connect to mongodb
client = MongoClient('localhost', 27017)
db = client['CompNets']
collection = db['SITC']

file_chunk = 5000

#with bz2.BZ2File("year_origin_destination_hs07_4.tsv.bz2", 'r') as fp:
with bz2.BZ2File("year_origin_destination_sitc_rev2.tsv.bz2", 'r') as fp:
#with open("test2.txt", 'r') as fp:
			'''

			'''
			i = 0
			data_list = []

			for line in fp:
				year, origin, dest, sitc, export_val, import_val = line.split("\t")

				if i == 0:
					i = i + 1
					continue

				#print import_val

				if import_val == 'NULL\n':
					import_val = 0

				if export_val == 'NULL':
					export_val = 0


				#print export_val
				#print import_val

				data = {'Year': int(year),
						'Origin': origin,
						'Destination': dest,
						'SITC': sitc,
						'Export': float(export_val),
						'Import': float(import_val)}
				#print data

				data_list.append(data)

				i = i + 1

				if i%file_chunk == 0:
					collection.insert_many(data_list)
					#print "inserted"
					data_list = []

			#print data_list
			collection.insert_many(data_list)
			fp.close()