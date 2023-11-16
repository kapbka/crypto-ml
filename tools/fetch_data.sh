set -ex

uri="mongodb://bot:sE]W.c<J~Me74dgE@ml.clrn.dev:27017/twitter"

source venv/bin/activate
mongoexport --uri "${uri}" --collection data_point | python ./tools/data_points_to_csv.py

