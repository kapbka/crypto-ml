db.getCollection("tweet").aggregate([
   { "$project": {"tokens": 1}},
   { "$unwind": "$tokens" },

   { "$group": { "_id": "$tokens", "count": { "$sum": 1} } },
   { "$group": {
      "_id": null,
         "counts": {
            "$push": {
               "k": "$_id",
               "v": "$count"
            }
         }
      } },
      { "$replaceRoot": {
         "newRoot": { "$arrayToObject": "$counts" }
   } }
])
