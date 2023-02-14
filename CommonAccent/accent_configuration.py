"""
Data preparation of CommonAccent dataset for Accent classification (English).
(Recipe that uses CV version 11.0)
Download: https://commonvoice.mozilla.org/en/datasets

Author
------
 * Juan Pablo Zuluaga 2023
"""
_ACCENTS_EN = [
    "Austrian", # 104
    "East African Khoja", # 107
    "Dutch", # 108
    "West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)", # 282
    "Welsh English", # 623
    "Malaysian English", # 1004
    "Liverpool English,Lancashire English,England English", # 2571
    "Singaporean English", # 2792
    "Hong Kong English", # 2951
    "Filipino", # 4030
    "Southern African (South Africa, Zimbabwe, Namibia)", # 4270
    "New Zealand English", # 4960
    "Irish English", # 6339
    "Northern Irish", # 6862
    "Scottish English", # 10817
    "Australian English", # 33335
    "German English,Non native speaker", # 41258
    "Canadian English", # 45640
    "England English", # 75772
    "India and South Asia (India, Pakistan, Sri Lanka)", # 79043
    "United States English", # 249284
]
_ACCENTS_FR = [
    "Français d’Algérie", # 319 
    "Français d’Allemagne", # 355 
    "Français du Bénin", # 823 
    "Français de La Réunion", # 884 
    "Français des États-Unis", # 898 
    "Français de Suisse", # 3608 
    "Français de Belgique", # 6509 
    "Français du Canada", # 8073 
    "Français de France", # 342921 
]
_ACCENTS_DE = [
    "Italienisch Deutsch", # 947 
    "Schweizerdeutsch", # 9891 
    "Österreichisches Deutsch", # 16066 
    "Nordrhein-Westfalen,Bundesdeutsch, Hochdeutsch,Deutschland Deutsch", # 50843 
    "Deutschland Deutsch", # 252709 
]
_ACCENTS_IT = [
    "Emiliano", # 151
    "Meridionale", # 193
    "Veneto", # 1508
    "Tendente al siculo, ma non marcato", # 2175
    "Basilicata,trentino", # 2297
]
_ACCENTS_ES = [
    "España: Islas Canarias", # 1326
    "Chileno: Chile, Cuyo", # 4285
    "América central", # 6031
    "Caribe: Cuba, Venezuela, Puerto Rico, República Dominicana, Panamá, Colombia caribeña, México caribeño, Costa del golfo de México", # 8329
    "España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha)", # 8683
    "Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay", # 11162
    "Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina", # 12997
    "México", # 26136
    "España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca)", # 30588
    "España: Sur peninsular (Andalucia, Extremadura, Murcia)", # 38251
]

