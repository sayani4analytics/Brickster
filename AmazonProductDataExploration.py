# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks and GraphFrames 
# MAGIC An exploration of the two technologies using the [Amazon product co-purchasing network dataset](http://snap.stanford.edu/data/index.html#amazon).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion
# MAGIC
# MAGIC We each took different approaches to storing and reading the raw .txt files of our data. The first way is by using Azure Blob Storage and using Spark's built-in support for reading Azure Blob Storage files with an account access key. The other way to do it is by uploading data directly to Databricks and storing it in the [Databricks File System](https://learn.microsoft.com/en-us/azure/databricks/dbfs/). We explored both options!
# MAGIC
# MAGIC ### Connecting to data stored in Azure Blob Storage
# MAGIC Our raw data in the form of .txt files was first uploaded to Azure Blob Storage, where the files are securely stored. Azure Blob Storage is Azure's generic storage solution, and has the advantage of high scalability and availability and not being native to Databricks. It is likely that an organization on the cloud may already have files of interest stored in Blob Storage somewhere. Although this was done using Azure Blob Storage, a similar solution would exist for referencing data hosted in other clouds such as on AWS in S3 buckets.
# MAGIC

# COMMAND ----------

# Declare storage account information
storage_account_name = "amazonproductdata"
# TODO: don't store storage key here...
storage_account_access_key = "<Storage Key Here>"

# Set up connection
file_type = "csv"
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# Define file paths
early_march_file_location = "wasbs://raw-data@amazonproductdata.blob.core.windows.net/amazon0302.txt"
late_march_file_location = "wasbs://raw-data@amazonproductdata.blob.core.windows.net/amazon0312.txt"
may_file_location = "wasbs://raw-data@amazonproductdata.blob.core.windows.net/amazon0505.txt"
june_file_location = "wasbs://raw-data@amazonproductdata.blob.core.windows.net/amazon0601.txt"
meta_file_location = "wasbs://raw-data@amazonproductdata.blob.core.windows.net/amazon-meta.txt"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connecting to data stored in the Databricks File System (DBFS)
# MAGIC Data can also be uploaded within the Databricks workspace UI to DBFS. This is convenient for many users and allows data to be uploaded and managed all in the same system. If flexibility or existing data are not concerns, this is a good option for getting started quickly without worrying about things like authentication to read files.

# COMMAND ----------

# TODO: file path declarations, overwriting azure storage ones

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading in the data and parsing it
# MAGIC Regardless of how the data was stored, we can simply reference the raw data files and parse them into DataFrames. DataFrames allow us to make the data tabular and manupulate the data with it intuitive APIs. 
# MAGIC
# MAGIC Like with storage, we each came up with slightly different ways to parse the edge data. Both are valid and show the flexibility of the Spark APIs!
# MAGIC
# MAGIC #### Reading edge data (using StructType)

# COMMAND ----------

# Edges
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Change to string and rename to src, dst, add time added? 
conn_schema = StructType([
    StructField("FromNodeId", IntegerType(), True),
    StructField("ToNodeId", IntegerType(), True)
])

early_march_df = spark.read.format(file_type) \
    .option("delimiter", "\t") \
    .option("comment", "#") \
    .schema(conn_schema) \
    .load(early_march_file_location)

late_march_df = spark.read.format(file_type) \
    .option("delimiter", "\t") \
    .option("comment", "#") \
    .schema(conn_schema) \
    .load(late_march_file_location)

may_df = spark.read.format(file_type) \
    .option("delimiter", "\t") \
    .option("comment", "#") \
    .schema(conn_schema) \
    .load(may_file_location)

june_df = spark.read.format(file_type) \
    .option("delimiter", "\t") \
    .option("comment", "#") \
    .schema(conn_schema) \
    .load(june_file_location)

display(early_march_df)
display(late_march_df)
display(may_df)
display(june_df)

edge_all_df=mar_2_df.union(mar_12_df).union(may_5_df).union(june_1_df)
edge_all_df.count()
display(edge_all_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading edge data (using map)

# COMMAND ----------

# TODO: add reading edge data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading product detail data (vertices)
# MAGIC Reading in the vertex data was more involved because the Amazon product metadata file did not come in a standard format like JSON or YAML. Luckily for us, the data was at least in a standard format. We first loaded each product into its own row, and then used a complex regular expression to extract the properties we cared about. On the initial read, splitting each product into its own row was important to avoid RAM limitations.

# COMMAND ----------

# Node metadata
from pyspark.sql.functions import col
from pyspark.sql import functions as F

regex = r'Id:\s*(\d+)\s*\nASIN:\s+(\w+)\s*\n\s*title:\s*([\w\W]+)\s*\r\n\s*group:\s+(.+)\s*\n\s*salesrank:\s*(-*\d+)\s*\n\s*similar:\s+(\d+)\s*(\d*.*?)\s*\n\s*categories:\s*(\d+)\s*\n\s*([\s\S]*)reviews:\s+total:\s+(\d+)\s+downloaded:\s+(\d+)\s+avg rating:\s+(\d+\.*\d*)\s*\n\s*([\s\S]*)'
df1 = spark.read.text(meta_file_location, lineSep="\r\n\r\n")
metadata_df = df1 \
        .withColumn("id", F.regexp_extract("value", r'(\d+)\s*\nASIN:\s+(\w+)\s*\n', 1)) \
        .withColumn("asin", F.regexp_extract("value", r'(\d+)\s*\nASIN:\s+(\w+)\s*\n', 2)) \
        .withColumn("title", F.regexp_extract("value", regex, 3)) \
        .withColumn("group", F.regexp_extract("value", regex, 4)) \
        .withColumn("salesrank", F.regexp_extract("value", regex, 5)) \
        .withColumn("similar_count", F.regexp_extract("value", regex, 6)) \
        .withColumn("similar", F.regexp_extract("value", regex, 7)) \
        .withColumn("categories_count", F.regexp_extract("value", regex, 8)) \
        .withColumn("categories", F.regexp_extract("value", regex, 9)) \
        .withColumn("reviews_total", F.regexp_extract("value", regex, 10)) \
        .withColumn("reviews_downloaded", F.regexp_extract("value", regex, 11)) \
        .withColumn("reviews_avg_rating", F.regexp_extract("value", regex, 12)) \
        .withColumn("reviews", F.regexp_extract("value", regex, 13)) \
        .withColumn("similar", F.split(F.col("similar"), "\s+").cast("array<string>")) \
        .withColumn("categories", F.split(F.col("categories"), "\r\n").cast("array<string>")) \
        .withColumn("reviews", F.split(F.col("reviews"), "\r\n").cast("array<string>")) \
        .where((F.col("id").isNotNull()) & (F.col("id") != ""))

display(metadata_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving Product Detail DataFrame to Delta Table
# MAGIC Although we first started with the DataFrame as-is, we found that some algorithms were very slow to run on just this data. Saving the parsed data into a Delta Table and then referencing the Delta Table values was a simple way to get a ~20x performance boosts (and better traceability for any changed values!).

# COMMAND ----------

# Save dataframe as delta table
metadata_df.write.saveAsTable("product_metadata")
edge_all_df.write.saveAsTable("amazon_edges")

# COMMAND ----------

# Reference delta table for dataframe 
node_metadata_df = spark.table("product_metadata")

# Visualizing the graph

import networkx as nx
#import matplotlib as plt

def PlotGraph(edge_list):
    Gplot=nx.Graph()
    for row in edge_list.select('src','dst').take(50):
        Gplot.add_edge(row['src'],row['dst'])

    #plt.subplot(121)
    nx.draw(Gplot, with_labels=True#, font_weight='bold'
            )

PlotGraph(g.edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Answering Query Questions
# MAGIC As part of the assignment, there were 6 questions to be answered from this dataset. The queries below answer the 6 questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q1: What are the percentages of each rating digit for the product with id: 21?
# MAGIC
# MAGIC #### Answer:
# MAGIC | Rating | Count | Percentage |
# MAGIC |--------|-------|------------|
# MAGIC | 5      | 100   | 71.43%     |
# MAGIC | 4      | 30    | 21.43%     |
# MAGIC | 3      | 3     | 2.14%      |
# MAGIC | 2      | 3     | 2.14%      |
# MAGIC | 1      | 4     | 2.86%      |
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

# Filter the DataFrame to only include rows where the id column is equal to 21
product_21_df = node_metadata_df.filter(F.col("id") == "21")

# Explode each review entry into multiple rows and extract rating
product_21_reviews_df = product_21_df.select(F.explode("reviews").alias("review"))
product_21_reviews_df = product_21_reviews_df.withColumn("rating", F.regexp_extract("review", r"rating:\s+(\d+)", 1))

# Count the number of occurrences of each rating
rating_counts_df = product_21_reviews_df.groupBy("rating").count()

# Calculate the percentage of each rating
total_ratings = product_21_df.select("reviews_total").collect()[0][0]
rating_percentages_df = rating_counts_df.withColumn("percentage", (F.col("count") / total_ratings) * 100)

# Sort the result by the rating column in descending order
rating_percentages_df = rating_percentages_df.orderBy(F.col("rating").desc())

# Display the result
display(rating_percentages_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2: Which pairs of products have consistently stayed within each other’s “Customers who bought this item also bought” lists through march, may and June of 2003?
# MAGIC
# MAGIC #### Answer:
# MAGIC There are 1,346 products that stayed consistently within each other's “Customers who bought this item also bought” lists.
# MAGIC
# MAGIC See the `consistent_bidirectional_edges` DataFrame for exact customer Id's.

# COMMAND ----------

from pyspark.sql import functions as F

# Create a temporary view for each DataFrame
early_march_df.createOrReplaceTempView("early_march")
late_march_df.createOrReplaceTempView("late_march")
may_df.createOrReplaceTempView("may")
june_df.createOrReplaceTempView("june")

# Find bidirectional edges in each DataFrame
bidirectional_edges_early_march = spark.sql("""
    SELECT DISTINCT e1.FromNodeId, e1.ToNodeId
    FROM early_march e1
    JOIN early_march e2
    ON e1.FromNodeId = e2.ToNodeId AND e1.ToNodeId = e2.FromNodeId
""")

bidirectional_edges_late_march = spark.sql("""
    SELECT DISTINCT l1.FromNodeId, l1.ToNodeId
    FROM late_march l1
    JOIN late_march l2
    ON l1.FromNodeId = l2.ToNodeId AND l1.ToNodeId = l2.FromNodeId
""")

bidirectional_edges_may = spark.sql("""
    SELECT DISTINCT m1.FromNodeId, m1.ToNodeId
    FROM may m1
    JOIN may m2
    ON m1.FromNodeId = m2.ToNodeId AND m1.ToNodeId = m2.FromNodeId
""")

bidirectional_edges_june = spark.sql("""
    SELECT DISTINCT j1.FromNodeId, j1.ToNodeId
    FROM june j1
    JOIN june j2
    ON j1.FromNodeId = j2.ToNodeId AND j1.ToNodeId = j2.FromNodeId
""")

# Find bidirectional edges that are present in all DataFrames
consistent_bidirectional_edges = bidirectional_edges_early_march \
 .intersect(bidirectional_edges_late_march) \
 .intersect(bidirectional_edges_may) \
 .intersect(bidirectional_edges_june)

# Display the result
display(consistent_bidirectional_edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3: For a product “A” in group “G”, at what level do you find a product of a different group in the June 1st dataset? 
# MAGIC
# MAGIC For any user specified product A (choose 1 random product) belonging to group G, if you consider the “Customers who bought this item also bought” list of June 01 to be level 0 and for each of the items in that list if you consider their “Customers who bought this item also bought” list of June 01 except A as level 1, then for each of those product’s own “Customers who bought this item also bought” list of June 01 except A as level 2 and so on, at what level do you find a product of a different group? Show the chain of products and their groups that were traversed, till you reach a different group. 
# MAGIC
# MAGIC
# MAGIC Levels explained:
# MAGIC ```
# MAGIC Level 0: A : { “Customers who bought this item also bought” list of June 01: [B, C]}
# MAGIC 	Level 1: B: {“Customers who bought this item also bought” list of June 01: [A, D, E]}
# MAGIC 	Level 1: C: {“Customers who bought this item also bought” list of June 01: [A, F, G]}
# MAGIC 		Level 2: D: {“Customers …” list of June 01: [A, H, I]}
# MAGIC 		Level 2: E: {“Customers …” list of June 01: [A, J, K]}
# MAGIC 		Level 2: F: {“Customers …” list of June 01: [A, L, M]}
# MAGIC 		Level 2: G: {“Customers …” list of June 01: [A, N, O]}
# MAGIC 			Level 3: H
# MAGIC 			Level 3: I
# MAGIC 			.
# MAGIC 			.
# MAGIC 			Level 3: O
# MAGIC 		Results expected:
# MAGIC 			Level 0: Product A: Group 1
# MAGIC 			Level 1: Product B: Group 1, Product C: Group 1
# MAGIC 			Level 2: Product D: Group 1, Product E: Group 1 … 
# MAGIC 			.
# MAGIC 			.
# MAGIC 			Level n:		.	.	 Product n: Group 2
# MAGIC ```
# MAGIC
# MAGIC #### Answer:
# MAGIC For two products I explored, both were found to have related items at a different group at level 1.
# MAGIC
# MAGIC ```
# MAGIC "The Casebook of Sherlock Holmes, Volume 2 (Casebook of Sherlock Holmes)" (id: 24, category: Book) =>  "Jonny Quest - Bandit in Adventures Best Friend" (id: 71, category: Video)
# MAGIC ```
# MAGIC And
# MAGIC ```
# MAGIC "Life Application Bible Commentary: 1 and 2 Timothy and Titus" (id: 4, category: Book) => "The NBA's 100 Greatest Plays" (id: 44, category: DVD)
# MAGIC ```

# COMMAND ----------

# Create graph
from graphframes import *

edges = june_df.selectExpr("FromNodeId as src", "ToNodeId as dst")
g = GraphFrame(node_metadata_df, edges)

# COMMAND ----------

# Product 1 search
start_title = "The Casebook of Sherlock Holmes, Volume 2 (Casebook of Sherlock Holmes)"
start_node = g.vertices.filter(f"title = '{start_title}'")
start_node_group = start_node.select("group").collect()[0][0]
start_node_id = start_node.select("id").collect()[0][0]

paths = g.bfs(f"id = {start_node_id}", f"group != '{start_node_group}' AND group != ''", maxPathLength=3).limit(1)
paths.show()

# COMMAND ----------

# Product 2 search
start_title = "Life Application Bible Commentary: 1 and 2 Timothy and Titus"
start_node = g.vertices.filter(f"title = '{start_title}'")
start_node_group = start_node.select("group").collect()[0][0]
start_node_id = start_node.select("id").collect()[0][0]

paths = g.bfs(f"id = {start_node_id}", f"group != '{start_node_group}' AND group != ''", maxPathLength=5).limit(1)
paths.show()

# COMMAND ----------



# MAGIC %md
# MAGIC ### Q4: What is the longest cyclic path taken by a user that starts at the book “Jack and the Beanstalk” to return back to where they started?
# MAGIC Consider a user is visiting amazon.com and is viewing the book “Jack and the Beanstalk” and from then on is only using the links under “Customers who bought this item also bought” list to view other items, and after a certain number of clicks is back to the page for “Jack and the Beanstalk”. If we call this a cyclic path, what is the longest cyclic path taken by a user to return to the book “Jack and the Beanstalk”?
# MAGIC #### Answer:

%sql

select * from product_metadata
where title like '%Jack%Beanstalk%'
and group='Book'

df_book=g.vertices.filter((col('title').contains('Jack and the Beanstalk')) & (col('group')=='Book'))
display(df_book)

# function to find longest path

def find_longest_cyclic_path(graph, vertex):
    visited = set()
    stack = [(vertex, [vertex])]  # Stack stores tuples of (current_vertex, path)

    longest_path = []

    while stack:
        current_vertex, path = stack.pop()

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        neighbors = graph.find(f"(a)-[]->(b)").filter(f"a.id = '{current_vertex}'").select("b.id").collect()

        for neighbor in neighbors:
            if neighbor[0] == vertex and len(path) > 2:
                # Found a cyclic path
                if len(path) > len(longest_path):
                    longest_path = path[:]
            elif neighbor[0] not in visited:
                stack.append((neighbor[0], path + [neighbor[0]]))

    return longest_path

start_vertex = "273043"  # Replace with the desired vertex
longest_path = find_longest_cyclic_path(g, start_vertex)
longest_path

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q5: In June 2003, for all the products in group DVD, how many items (clicks) would it take for a customer to go from the bestselling product to the worst selling product? What would the path taken be?
# MAGIC In June 2003, for all the products in group DVD, if a user is at the amazon.com page for the bestselling product and from then on is only using the links under “Customers who bought this item also bought” list to view other items, how many items (clicks) later will they reach the worst selling product? What was the path taken? List the items in the order in which they were clicked. 
# MAGIC [NOTE: Salesrank of each product is calculated based on how many times it was sold. A product climbs up the rank as it’s sales increases. Therefore, lower the salesrank number higher it is in the order and number of sales]
# MAGIC
# MAGIC #### Answer:
# MAGIC

# Pseudo-code:
# 1. Set up a new graphframe with vertices data and June purchase edge data
# 2. Filter the graphframe data for product group "DVD"
# 3. Find the best selling product and worst selling product. Best selling product will have lower sales rank and worst selling product will have higher sales rank

# %%
# create graphframe for June purchases and a dataframe that filter down to product group "DVD"
vertices_df_dvd=vertices_df.filter(col("group")=="DVD")
g_june_dvd=GraphFrame(vertices_df_dvd,june_1_df)
g_june=GraphFrame(vertices_df,june_1_df)


display(g_june_dvd.vertices)
df_dvd=g_june_dvd.vertices

#
# Best selling product with least rank
best_prod=df_dvd.where((col('salesrank')!='-1') & (col('salesrank')!='0')).withColumn("rank_numeric",col('salesrank').cast("int")).orderBy("rank_numeric",ascending=True).head(1)
display(best_prod)

# %%
# Worst selling product with highest rank
worst_prod=df_dvd.where((col('salesrank')!='-1') & (col('salesrank')!='0')).withColumn("rank_numeric",col('salesrank').cast("int")).orderBy("rank_numeric",ascending=False).head(1)
display(worst_prod)

#
# Now that we have got the id of best and worst product, we need to find the paths connecting best to worst product

# Best product id= 193107
# Worst product id= 22358

paths_best_to_worst=g_june.shortestPaths(landmarks=['22358'])
paths_best_to_worst.select("id", "distances").filter(col('id')=='193107').show()


paths_best_to_worst_bfs=g_june.bfs("id=193107","id=22358")
paths_best_to_worst_bfs.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q6: For product with ASIN: 0385492081 display the most helpful review(s) of the highest rating and the most helpful review(s) of the lowest rating.
# MAGIC
# MAGIC #### Answer:

# - Filter the vertices dataframe for this ASIN and create a new dataframe.
# - Take the “reviews” column which is an array and unnest it using the explode function into new rows. Create a new reviews dataframe
# - Using regex, extract the rating , votes, helpful columns
# - Create a temporaryview using this new dataframe and analyze the data using SQL

from pyspark.sql.functions import explode
#Create a dataframe which is filtered for this ASIN
df_asin=vertices_df.filter(col("asin")=='0385492081').select("id","asin","title","reviews","reviews_total")


#Retrieve ratings, votes and helpful from the reviews column.
df_asin_reviews=df_asin.select(explode("reviews").alias("review"))


# Regex and extract the rating , votes, helpful columns 
df_asin_reviews=df_asin_reviews.withColumn("rating", F.regexp_extract("review", r"rating:\s+(\d+)", 1))\
                        .withColumn("votes", F.regexp_extract("review", r"votes:\s+(\d+)", 1))\
                        .withColumn("helpful", F.regexp_extract("review", r"helpful:\s+(\d+)", 1))
                        
df_asin_reviews_final=df_asin.join(df_asin_reviews).drop("reviews")

# Create temporaryview
df_asin_reviews_final.createOrReplaceTempView("df_asin_reviews")

%sql
select asin, review, rating, helpful, votes from df_asin_reviews

# %%
%sql
#highest and lowest rating
select 
  max(rating) as highest_rating,
  min(rating) as lowest_rating
from df_asin_reviews


# %%
%sql
#most helpful review with highest rating
select 
  review,
  helpful,
  rating
from df_asin_reviews
where  helpful= (select max(helpful) from df_asin_reviews) #most helpful review
and (rating=(select max(rating) as highest_rating from df_asin_reviews) #highest rating
OR rating=(select min(rating) as highest_rating from df_asin_reviews) #lowest rating
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GraphFrames Exploration
# MAGIC After answering those questions, we wanted to explore a little bit more. Specifically, we wanted to see what we could do with the GraphFrames APIs.

# COMMAND ----------

# Product recommendations
def recommend_products(product_id, n):
    # Find the neighbors of the given product
    neighbors = g.edges.filter(F.col("src") == product_id)
        #  | (F.col("dst") == product_id)) \
        # .select(F.when(F.col("src") == product_id, F.col("dst")).otherwise(F.col("src")).alias("neighbor"))
    
    # Join with the vertices DataFrame to get product details
    recommendations = neighbors.join(vertices, neighbors["dst"] == vertices["id"]) \
        .select("id", "title", "group", "salesrank")
    
    return recommendations

# Community detection
result = g.labelPropagation(maxIter=5)
communities = result.select("id", "label")

# PageRank
result = g.pageRank(resetProbability=0.15, maxIter=10)
page_ranks = result.vertices.select("id", "pagerank")

display(communities)
display(page_ranks)


# COMMAND ----------

top_20 = page_ranks.orderBy("pagerank", ascending=False).limit(20)
display(top_20)

# COMMAND ----------

display(top_20.join(node_metadata_df, "id"))

# COMMAND ----------

# Find the size of each community
community_sizes = communities.groupBy("label").count()

# Find the biggest communities + join with details
biggest_communities = community_sizes.orderBy(F.col("count").desc()).limit(10)
biggest_community_details = communities.join(node_metadata_df, "id").join(biggest_communities, "label")

# Find the smallest communities + join with details
smallest_communities = community_sizes.orderBy(F.col("count").asc()).limit(10)
smallest_community_details = communities.join(node_metadata_df, "id").join(smallest_communities, "label")


# COMMAND ----------

from pyspark.sql.window import Window

# Calculate the proportion of each group within each community
biggest_group_proportions = biggest_community_details.groupBy("label", "group").count() \
    .withColumn("proportion", F.col("count") / F.sum("count").over(Window.partitionBy("label")))
smallest_group_proportions = smallest_community_details.groupBy("label", "group").count() \
    .withColumn("proportion", F.col("count") / F.sum("count").over(Window.partitionBy("label")))


# Show the result
display(biggest_group_proportions)
display(smallest_group_proportions)

# COMMAND ----------

# Get average review rating and average number of ratings for communities
biggest_community_ratings = biggest_community_details.groupBy("label") \
    .agg(
        F.avg("reviews_avg_rating").alias("avg_review_rating"),
        F.avg("reviews_total").alias("avg_num_ratings")
    )

smallest_community_ratings = smallest_community_details.groupBy("label") \
    .agg(
        F.avg("reviews_avg_rating").alias("avg_review_rating"),
        F.avg("reviews_total").alias("avg_num_ratings")
    )

display(biggest_community_ratings)
display(smallest_community_ratings)
