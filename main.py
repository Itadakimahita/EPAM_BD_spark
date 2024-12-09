from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, expr, split, round, when
from pyspark.sql.types import StringType
import requests
import geohash2



restaurant_data_path = "./restaurant_csv/"
weather_data_path = "./weather/year=2016/month=10/day=01/"
opencage_api_key = "3cd1a00ddd76487b93102f83a39b63f2"

def fetch_coordinates(address):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={opencage_api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                location = data['results'][0]['geometry']
                return f"{location['lat']}, {location['lng']}"
        return "{}"
    except Exception as e:
        print(f"Error fetching coordinates: {e}")
        return "{}"

# Register UDF for geohash generation
@udf(StringType())
def generate_geohash(lat, lon):
    try:
        return geohash2.encode(lat, lon, precision=4)
    except Exception as e:
        return None



if __name__ == '__main__':

    spark = SparkSession.builder \
    .appName("RestaurantDataEnrichment") \
    .config("spark.master", "local") \
    .getOrCreate()

    # Load restaurant data
    restaurant_df = spark.read.csv(restaurant_data_path, header=True, inferSchema=True)
    restaurant_df.show(5, truncate=False)

    # Check and handle null latitude and longitude
    missing_coords_df = restaurant_df.filter(col("lat").isNull() | col("lng").isNull())
    missing_coords_df.show(5, truncate=False)

    # This part assumes missing coordinates are fetched based on an "address" column.
    missing_coords = missing_coords_df.rdd.map(lambda row: (row.id, fetch_coordinates(f"{row.franchise_name},+countycode={row.country},+{row.city}")))
    updated_coords_df = missing_coords.toDF(["id", "coordinates"])
    

    updated_coords_df = updated_coords_df.withColumn(
            "updated_lat",
            round(split(col("coordinates"), ", ")[0].cast("double"), 3)
        ).withColumn(
            "updated_lng",
            round(split(col("coordinates"), ", ")[1].cast("double"), 3)
        )

    # Drop the old `coordinates` column
    updated_coords_df = updated_coords_df.drop("coordinates")

    # Add updated latitude and longitude back to the original DataFrame
    enriched_restaurant_df = restaurant_df.join(
        updated_coords_df,
        on="id",
        how="left"
    ).withColumn(
        "lat",
        when(col("lat").isNull(), col("updated_lat")).otherwise(col("lat"))
    ).withColumn(
        "lng",
        when(col("lng").isNull(), col("updated_lng")).otherwise(col("lng"))
    ).drop("updated_lat", "updated_lng")


    enriched_restaurant_df = enriched_restaurant_df.withColumnRenamed("lat", "restaurant_lat").withColumnRenamed("lng", "restaurant_lng")

    # Generate geohash column for enriched_restaurant_df
    enriched_restaurant_df = enriched_restaurant_df.withColumn(
        "geohash",
        generate_geohash(col("restaurant_lat"), col("restaurant_lng"))
    )
    enriched_restaurant_df.show(10, truncate=False)

    # Load weather data
    weather_df = spark.read.parquet(weather_data_path)

    # Rename latitude and longitude columns for weather_df
    weather_df = weather_df.withColumnRenamed("lat", "weather_lat").withColumnRenamed("lng", "weather_lng")

    # Generate geohash column for weather_df
    weather_df = weather_df.withColumn(
        "geohash",
        generate_geohash(col("weather_lat"), col("weather_lng"))
    )
    weather_df.show(10, truncate=False)

    # Join restaurant and weather data
    joined_df = enriched_restaurant_df.join(
        weather_df,
        on="geohash",
        how="left"
    )

    # Store result in Parquet format
    output_path = "./result"
    joined_df.write.mode("overwrite").parquet(output_path)
    spark.read.parquet(output_path).show(10, truncate=False)

    print("Data enrichment and storage complete!")
