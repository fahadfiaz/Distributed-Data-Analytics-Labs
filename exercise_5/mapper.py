import sys
column_name = [ "FL_DATE" , "OP_UNIQUE_CARRIER" , "OP_CARRIER_FL_NUM" , "ORIGIN" , "DEST" , "DEP_TIME" , "DEP_DELAY" , "ARR_TIME" , "ARR_DELAY" ]
for line in sys.stdin:
   line = line.split( ',' )
   try: 
       print((line[3],line[6],line[8]))
   except Exception as e:
       print("Error: ",e)
       
