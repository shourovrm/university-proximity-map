# Finding out US R1 & R2 Universities in Close Proximity

## Mount Drives

This is done to mount the Google Drive for Google Colab.


```python
# ### Mount google drive
# from google.colab import drive
# drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
# ## set working directory
# import os
# os.chdir('/content/drive/MyDrive/Office/Dropbox/RMS/Research/Grad-Application-2026/Data-analysis')
```

## Import Modules and files


```python
# for xlsx
! pip install openpyxl 
```

    Requirement already satisfied: openpyxl in c:\users\shour\miniconda3\envs\ds\lib\site-packages (3.1.5)
    Requirement already satisfied: et_xmlfile in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from openpyxl) (2.0.0)
    


```python
# for map
!pip install folium 
```

    Requirement already satisfied: folium in c:\users\shour\miniconda3\envs\ds\lib\site-packages (0.20.0)
    Requirement already satisfied: branca>=0.6.0 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from folium) (0.8.2)
    Requirement already satisfied: jinja2>=2.9 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from folium) (3.1.6)
    Requirement already satisfied: numpy in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from folium) (2.3.4)
    Requirement already satisfied: requests in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from folium) (2.32.5)
    Requirement already satisfied: xyzservices in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from folium) (2025.10.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from jinja2>=2.9->folium) (3.0.3)
    Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from requests->folium) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from requests->folium) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from requests->folium) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\shour\miniconda3\envs\ds\lib\site-packages (from requests->folium) (2025.11.12)
    


```python
import pandas as pd
import numpy as np

```

## R1 and R2 Universities

The Carnegie Classfication Database is downloaded from https://carnegieclassifications.acenet.edu/institutions/?inst=&research2025%5B%5D=1&research2025%5B%5D=2.

I have selected only the R1 and R2 universities.


```python
# Load Excel file of the carnegie classification file
## downloaded from https://carnegieclassifications.acenet.edu/institutions/?inst=&research2025%5B%5D=1&research2025%5B%5D=2&research2025%5B%5D=3#
#r1r2_info = pd.read_excel("R1R2-info.xlsx")
r1r2_info = pd.read_csv("ace-institutional-classifications.csv", low_memory=False)
```


```python
df = r1r2_info
```


```python
# Show all column names
df.info()
```


```python
df.head()
```

### Convert research2025name into only "R1" or "R2"



```python
# Clean and normalize the text
df["research_clean"] = (
    df["Research Activity Designation"]
        .astype(str)
        .str.strip()              # remove leading/trailing spaces
        .str.normalize('NFKD')    # remove hidden unicode variations
        .str.replace(r'\s+', ' ', regex=True)   # force single spaces
)

```


```python
print(df["research_clean"].unique())
```

    ['Research 2: High Research Spending and Doctorate Production'
     'Research 1: Very High Research Spending and Doctorate Production']
    


```python
# Convert research2025name into only "R1" or "R2"

# Option 1: overwrite the existing column
df["Research Activity Designation"] = np.where(
    df["Research Activity Designation"].str.contains("Research 1", na=False),
    "R1",
    np.where(
        df["Research Activity Designation"].str.contains("Research 2", na=False),
        "R2",
        None
    )
)

print(df["Research Activity Designation"].value_counts(dropna=False))

```

    Research Activity Designation
    R1    187
    R2    139
    Name: count, dtype: int64
    


```python
# Clean and normalize the text
df["instnm_clean"] = (
    df["name"]
        .astype(str)
        .str.strip()              # remove leading/trailing spaces
        .str.normalize('NFKD')    # remove hidden unicode variations
        .str.replace(r'\s+', ' ', regex=True)   # force single spaces
)
```


```python
df["name"] = df["instnm_clean"]
df = df.drop(columns=["instnm_clean"])   # optional: remove helper column
```


```python
df.head()
```


```python
r1r2 = df[['unitid','name', 'city', 'Research Activity Designation', 'state']]
r1r2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unitid</th>
      <th>name</th>
      <th>city</th>
      <th>Research Activity Designation</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>222178</td>
      <td>Abilene Christian University</td>
      <td>Abilene</td>
      <td>R2</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200697</td>
      <td>Air Force Institute of Technology-Graduate Sch...</td>
      <td>Wright-Patterson AFB</td>
      <td>R2</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>2</th>
      <td>385415</td>
      <td>Albert Einstein College of Medicine</td>
      <td>Bronx</td>
      <td>R2</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>131159</td>
      <td>American University</td>
      <td>Washington</td>
      <td>R1</td>
      <td>DC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197869</td>
      <td>Appalachian State University</td>
      <td>Boone</td>
      <td>R2</td>
      <td>NC</td>
    </tr>
  </tbody>
</table>
</div>




```python
# keep only these columns
r1r2 = r1r2.rename(columns={"Research Activity Designation": "R1/R2", "name": "institutes", "state": "states"})
r1r2.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unitid</th>
      <th>institutes</th>
      <th>city</th>
      <th>R1/R2</th>
      <th>states</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>222178</td>
      <td>Abilene Christian University</td>
      <td>Abilene</td>
      <td>R2</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200697</td>
      <td>Air Force Institute of Technology-Graduate Sch...</td>
      <td>Wright-Patterson AFB</td>
      <td>R2</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>2</th>
      <td>385415</td>
      <td>Albert Einstein College of Medicine</td>
      <td>Bronx</td>
      <td>R2</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>131159</td>
      <td>American University</td>
      <td>Washington</td>
      <td>R1</td>
      <td>DC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197869</td>
      <td>Appalachian State University</td>
      <td>Boone</td>
      <td>R2</td>
      <td>NC</td>
    </tr>
  </tbody>
</table>
</div>



## US Institutes Info

The database of US universities with geo information is downloaded from US Dept. of Education - https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-Institution_05192025.zip


```python
df2 = pd.read_csv("inst-cohort.csv", low_memory=False)

```


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6429 entries, 0 to 6428
    Columns: 3306 entries, UNITID to SCORECARD_SECTOR
    dtypes: float64(920), int64(14), object(2372)
    memory usage: 162.2+ MB
    


```python
print(df2.columns)
```

    Index(['UNITID', 'OPEID', 'OPEID6', 'INSTNM', 'CITY', 'STABBR', 'ZIP',
           'ACCREDAGENCY', 'INSTURL', 'NPCURL',
           ...
           'COUNT_WNE_MALE1_P11', 'GT_THRESHOLD_P11', 'MD_EARN_WNE_INC1_P11',
           'MD_EARN_WNE_INC2_P11', 'MD_EARN_WNE_INC3_P11',
           'MD_EARN_WNE_INDEP0_P11', 'MD_EARN_WNE_INDEP1_P11',
           'MD_EARN_WNE_MALE0_P11', 'MD_EARN_WNE_MALE1_P11', 'SCORECARD_SECTOR'],
          dtype='object', length=3306)
    


```python
##### Find out relevant columns
```


```python
# show columns that has the text private

cols_with_private = []

for col in df2.select_dtypes(include="object").columns:
    mask = df2[col].str.contains("private", case=False, na=False)
    if mask.any():
        cols_with_private.append(col)
        sample_value = df2.loc[mask, col].iloc[0]
        print(f"{col}: {sample_value}")

cols_with_private

```

    NPCURL: https://www.sscc.edu/_private/npcalc.htm
    CONTROL_PEPS: Private Nonprofit
    




    ['NPCURL', 'CONTROL_PEPS']




```python
print(df2["CONTROL_PEPS"].head())
```

    0               Public
    1               Public
    2    Private Nonprofit
    3               Public
    4               Public
    Name: CONTROL_PEPS, dtype: object
    


```python
# find any column containing lat
[col for col in df2.columns if "LAT" in col.upper()]
```




    ['LATITUDE']




```python
df2['STABBR'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STABBR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>




```python
# find any column containing lon
[col for col in df2.columns if "LON" in col.upper()]
```




    ['LONGITUDE']




```python
# keep only these columns
inst = df2[['UNITID','INSTNM', 'CITY', 'STABBR','LONGITUDE', 'LATITUDE', 'CONTROL_PEPS']]
inst.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UNITID</th>
      <th>INSTNM</th>
      <th>CITY</th>
      <th>STABBR</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>CONTROL_PEPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100654</td>
      <td>Alabama A &amp; M University</td>
      <td>Normal</td>
      <td>AL</td>
      <td>-86.568502</td>
      <td>34.783368</td>
      <td>Public</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100663</td>
      <td>University of Alabama at Birmingham</td>
      <td>Birmingham</td>
      <td>AL</td>
      <td>-86.799345</td>
      <td>33.505697</td>
      <td>Public</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100690</td>
      <td>Amridge University</td>
      <td>Montgomery</td>
      <td>AL</td>
      <td>-86.174010</td>
      <td>32.362609</td>
      <td>Private Nonprofit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100706</td>
      <td>University of Alabama in Huntsville</td>
      <td>Huntsville</td>
      <td>AL</td>
      <td>-86.640449</td>
      <td>34.724557</td>
      <td>Public</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100724</td>
      <td>Alabama State University</td>
      <td>Montgomery</td>
      <td>AL</td>
      <td>-86.295677</td>
      <td>32.364317</td>
      <td>Public</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename columns
inst = inst.rename(columns ={"INSTNM": "INSTITUTES", "STABBR": "STATES_ABB", "CONTROL_PEPS": "PUBLIC/PRIVATE"})
```

## Merge the Institute Info and R1/R2 Data


```python
inst.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6429 entries, 0 to 6428
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   UNITID          6429 non-null   int64  
     1   INSTITUTES      6429 non-null   object 
     2   CITY            6429 non-null   object 
     3   STATES_ABB      6429 non-null   object 
     4   LONGITUDE       5924 non-null   float64
     5   LATITUDE        5924 non-null   float64
     6   PUBLIC/PRIVATE  6405 non-null   object 
    dtypes: float64(2), int64(1), object(4)
    memory usage: 351.7+ KB
    


```python
r1r2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 326 entries, 0 to 325
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   unitid      326 non-null    int64 
     1   institutes  326 non-null    object
     2   city        326 non-null    object
     3   R1/R2       326 non-null    object
     4   states      326 non-null    object
    dtypes: int64(1), object(4)
    memory usage: 12.9+ KB
    


```python
# Select required columns from inst
inst_sel = inst[[
    "UNITID", "INSTITUTES", "CITY", "STATES_ABB",
    "LONGITUDE", "LATITUDE", "PUBLIC/PRIVATE"
]].drop_duplicates(subset="UNITID")

```


```python
# Merge using UNITID
merged = r1r2.merge(
    inst_sel,
    left_on="unitid",
    right_on="UNITID",
    how="left"
)

```


```python
# Keep only the needed columns, in the required order

merged = merged[[
    "UNITID",        # from inst
    "INSTITUTES",
    "CITY",
    "states",        # from r1r2
    "R1/R2",         # from r1r2
    "PUBLIC/PRIVATE",
    "LONGITUDE",
    "LATITUDE"
]]

```


```python
# remove "the" from the university names

merged["INSTITUTES"] = (
    merged["INSTITUTES"]
        .str.replace(r"^the\s+", "", case=False, regex=True)
        .str.strip()
)
```

## CS Ranking Data

Computer Science open rankings compiled by Brown University is collected in a csv file.

https://drafty.cs.brown.edu/csopenrankings/


```python

# Load the file
cs = pd.read_csv("csbrownrank.csv", low_memory=False)
```


```python
cs["university"].head()
```




    0                    Carnegie Mellon University+
    1         Massachusetts Institute of Technology+
    2            University of California, Berkeley+
    3                           Stanford University+
    4    University of Illinois at Urbana-Champaign+
    Name: university, dtype: object




```python
cs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 230 entries, 0 to 229
    Data columns (total 8 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   #                  230 non-null    int64  
     1   university         230 non-null    object 
     2   size               124 non-null    float64
     3   U.S. News          230 non-null    int64  
     4   csrankings.org     201 non-null    float64
     5   placement rank     191 non-null    float64
     6   best paper awards  121 non-null    float64
     7   total              230 non-null    int64  
    dtypes: float64(4), int64(3), object(1)
    memory usage: 14.5+ KB
    


```python
# Rename "#" to "rank"
cs = cs.rename(columns={"#": "rank"})
```


```python
# Clean the university names (strip spaces, quotes, special chars)
cs["university"] = (
    cs["university"]
    .astype(str)
    .str.strip()  # # remove leading/trailing spaces
    .str.replace(r"[^\w\s.&-]", "", regex=True)   # remove strange characters
    .str.replace(r"^the\s+", "", case=False, regex=True)
)
```

### Merge those matched


```python
# Check how many names match
matches = cs["university"].isin(merged["INSTITUTES"]).sum()

print("Total CS universities:", len(cs))
print("Total matches with R1/R2 list:", matches)
```

    Total CS universities: 230
    Total matches with R1/R2 list: 138
    


```python
## Merge 'merged' (R1/R2 table) with CS rank

merged_cs = merged.merge(
    cs[["university", "rank"]],
    left_on="INSTITUTES",
    right_on="university",
    how="left"
)
```


```python
merged_cs["rank"].notna().sum()
```




    np.int64(138)




```python
# merged_cs.to_excel("final-merged.xlsx", index=False) ## match this with the unmatched values of csrank.
```

### Check unmatched data


```python
# Identify unmatched rows from merged_cs

non_matched = cs[~cs["university"].isin(merged["INSTITUTES"])]

```


```python
print("Total universities in CS ranking:", len(cs))
print("Matched:", cs["university"].isin(merged["INSTITUTES"]).sum())
print("Not matched:", len(non_matched))

```

    Total universities in CS ranking: 230
    Matched: 138
    Not matched: 92
    


```python
non_matched.to_excel("cs_non_matched.xlsx", index=False)

```

### Merge the fixed data

The cs_non_matched.xlsx file has been updated with corrected university names that match those in final-merged.xlsx. To be noted that some universities in the CS ranking are not really R1 or R2 institutions.


```python
cs_fixed = pd.read_excel("cs_unmatched.xlsx")
```


```python
# Merge only to pull the corrected ranks

temp = merged_cs.merge(
    cs_fixed[["university", "rank"]],
    left_on="INSTITUTES",
    right_on="university",
    how="left"
)

```


```python
# Update the existing rank column
temp["rank"] = temp["rank_x"].fillna(temp["rank_y"])
```


```python
# remove helper columns
final = temp.drop(columns=["rank_x", "rank_y", "university_x", "university_y"])
```


```python
# write the NaN as None
final["rank"] = final["rank"].fillna("N/A")


```


```python
final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UNITID</th>
      <th>INSTITUTES</th>
      <th>CITY</th>
      <th>states</th>
      <th>R1/R2</th>
      <th>PUBLIC/PRIVATE</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>222178</td>
      <td>Abilene Christian University</td>
      <td>Abilene</td>
      <td>TX</td>
      <td>R2</td>
      <td>Private Nonprofit</td>
      <td>-99.709797</td>
      <td>32.468943</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200697</td>
      <td>Air Force Institute of Technology-Graduate Sch...</td>
      <td>Wright-Patterson AFB</td>
      <td>OH</td>
      <td>R2</td>
      <td>Public</td>
      <td>-84.082618</td>
      <td>39.782221</td>
      <td>191.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>385415</td>
      <td>Albert Einstein College of Medicine</td>
      <td>Bronx</td>
      <td>NY</td>
      <td>R2</td>
      <td>Private Nonprofit</td>
      <td>-73.846327</td>
      <td>40.852847</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>131159</td>
      <td>American University</td>
      <td>Washington</td>
      <td>DC</td>
      <td>R1</td>
      <td>Private Nonprofit</td>
      <td>-77.088875</td>
      <td>38.936005</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197869</td>
      <td>Appalachian State University</td>
      <td>Boone</td>
      <td>NC</td>
      <td>R2</td>
      <td>Public</td>
      <td>-81.680583</td>
      <td>36.215536</td>
      <td>N/A</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check how many updates

before = merged_cs["rank"].notna().sum()
after  = final["rank"].notna().sum()

print("Before:", before)
print("After :", after)
print("Newly filled ranks:", after - before)


```

    Before: 138
    After : 326
    Newly filled ranks: 188
    


```python
# final.to_excel("final_with_fixed_ranks.xlsx", index=False)
```

### still missing rank


```python
# Filter rows where rank is missing
missing_rank = final[final["rank"].isna()].copy()

# Count how many R1 and R2 have no ranking
missing_counts = missing_rank["R1/R2"].value_counts()

print("Missing R1 rankings:", missing_counts.get("R1", 0))
print("Missing R2 rankings:", missing_counts.get("R2", 0))

```

    Missing R1 rankings: 0
    Missing R2 rankings: 0
    


```python
# # 3. Save to Excel
# missing_rank.to_excel("missing_rank_universities.xlsx", index=False)
```

## Find Closest Universities


```python
from geopy.distance import geodesic
import itertools
import pandas as pd
```


```python
# Keep valid coordinates
df_coords = final.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

pairs = []

for (i1, row1), (i2, row2) in itertools.combinations(df_coords.iterrows(), 2):

    dist_km = geodesic(
        (row1["LATITUDE"], row1["LONGITUDE"]),
        (row2["LATITUDE"], row2["LONGITUDE"])
    ).km

    pairs.append([
        row1["UNITID"], row1["INSTITUTES"], row1["PUBLIC/PRIVATE"], row1["CITY"], row1["states"], row1["R1/R2"], row1["rank"],
        row2["UNITID"], row2["INSTITUTES"], row2["PUBLIC/PRIVATE"], row2["CITY"], row2["states"], row2["R1/R2"], row2["rank"],
        dist_km
    ])

# Create dataframe with proper column names
distance_df = pd.DataFrame(
    pairs,
    columns=[
        "UNITID_1", "University_1", "Type_1", "City_1", "State_1", "R_Type_1", "CS_Rank_1",
        "UNITID_2", "University_2", "Type_2", "City_2", "State_2", "R_Type_2", "CS_Rank_2",
        "Distance_km"
    ]
)

# Sort and filter
dist_50 = (
    distance_df
        .sort_values(by="Distance_km")
        .query("Distance_km <= 50")
        .reset_index(drop=True)
)

```


```python
# Save output
dist_50.to_csv("close-universities.csv", index=False)
```


```python
dist_50.head()
```

## Cluster Map


```python
from folium.plugins import MarkerCluster, HeatMap
import folium
import numpy as np

```


```python
### Prepare data
# Clean rank column -> convert to numeric
df_geo["rank"] = pd.to_numeric(df_geo["rank"], errors="coerce")

# df_geo = final.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()


# Normalize rank (lower rank = better → higher normalized value)
ranks = df_geo["rank"]
rank_norm = (ranks.max() - ranks) / (ranks.max() - ranks.min())

# Fill missing ranks with mid-value 0.5
df_geo["rank_norm"] = rank_norm.fillna(0.5)


```


```python
# convert normalized rank to color

def rank_to_color(x):
    # green → yellow → red
    r = int(255 * x)
    g = int(255 * (1 - x))
    b = 60
    return f"#{r:02x}{g:02x}{b:02x}"

```


```python
# create map

m = folium.Map(location=[39.5, -98.35], zoom_start=4)

cluster = MarkerCluster().add_to(m)

```


```python
# color for R1 and R2

def rtype_color(rtype):
    if rtype == "R1":
        return "#1f77b4"   # blue
    elif rtype == "R2":
        return "#d62728"   # red
    else:
        return "#888888"   # fallback

```


```python
# plot markers

for _, row in df_geo.iterrows():
    color = rtype_color(row["R1/R2"])

    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.9,
        popup=(
            f"<b>{row['INSTITUTES']}</b><br>"
            f"State: {row['states']}<br>"
            f"Type: {row['PUBLIC/PRIVATE']}<br>"
            f"R-Type: {row['R1/R2']}<br>"
            f"CS Rank: {row['rank']}"
        )
    ).add_to(cluster)

```


```python
# Add Heatmap (for density)

heat_data = df_geo[["LATITUDE", "LONGITUDE"]].values.tolist()
HeatMap(heat_data, radius=18, blur=12).add_to(m)

```




    <folium.plugins.heat_map.HeatMap at 0x1e9844d5410>




```python
coord_lookup = final.set_index("UNITID")[["LATITUDE", "LONGITUDE"]].to_dict("index")
```


```python
## Distance lines

from folium import FeatureGroup

line_group = FeatureGroup(name="Distance Lines").add_to(m)

for _, row in dist_50.iterrows():
    lat1 = coord_lookup[row["UNITID_1"]]["LATITUDE"]
    lon1 = coord_lookup[row["UNITID_1"]]["LONGITUDE"]
    lat2 = coord_lookup[row["UNITID_2"]]["LATITUDE"]
    lon2 = coord_lookup[row["UNITID_2"]]["LONGITUDE"]

    line = folium.PolyLine(
        [(lat1, lon1), (lat2, lon2)],
        color="blue",
        weight=2,
        tooltip=f"{row['Distance_km']:.1f} km"
    )

    # store distance inside leaflet object
    line.add_child(folium.Popup(str(row["Distance_km"])))
    line_group.add_child(line)

```



```python
# show map

m
```


```python
m.save("../index.html")
```
