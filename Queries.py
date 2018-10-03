#!/usr/bin/python2

from pymongo import MongoClient
import json
import csv

# connect to mongodb
client = MongoClient('localhost', 27017)
db = client['CompNets']
collection = db['SITC']

year_list = [1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2014]

#year_list = [1965]

for number in year_list:

	cursor = collection.aggregate([{'$match': {'Year': number}}, {'$group': {'_id':{'Origin': '$Origin'}, 'TotalExport': {'$sum': '$Export'}, 'TotalImport': {'$sum': '$Import'}}}])

	filename = 'data_tot_trade_'+str(number)+'.json'
	
	with open(filename, 'w') as f:
		key = 0;
		data = {};
		
		for document in cursor:
	
			#json.dump(document, f)
			year = document.get('_id').get('Year')
			origin = document.get('_id').get('Origin')
			destination = document.get('_id').get('Destination')
			tot_export = document.get('TotalExport')
			tot_import = document.get('TotalImport')

			record = {'Year': year, 'Origin': origin, 'Destination': destination, 'Export': tot_export, 'Import': tot_import}
			#f.write(str(record))
			#f.write('\n')
			data[key] = record;
			key = key + 1;
		json.dump(data,f);