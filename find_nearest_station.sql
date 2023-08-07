-- Using the latitude and longitude of each event and weather station,
-- this query calculates the distance between each one and returns the 
-- closest weather station to each event. Note: some stations changed
-- their name over time and so have multiple entries, equidistant.


-- Write output to csv directly
\o '/home/david/Documents/ARU/AvalancheProject/demo/data/nearest_stations.csv'
\timing
-- The meta-commands \a , \f , and \pset footer 
-- for unaligned, comma-separated data with no footer.
\a \f , \pset footer

SELECT 
  av_sites.ob_date, 
  station_inv.station_name, 
  ST_Distance(av_sites.geog_point, station_inv.geog_point)/1000 as "distance(km)",
  station_id
FROM 
  av_sites 
  CROSS JOIN station_inv
  JOIN (
    SELECT 
      ob_date, 
      MIN(ST_Distance(av_sites.geog_point, station_inv.geog_point)) as min_distance
    FROM 
      av_sites 
      CROSS JOIN station_inv 
    WHERE 
      ST_Distance(av_sites.geog_point, station_inv.geog_point) IS NOT NULL
      AND EXTRACT(YEAR FROM av_sites.ob_date) BETWEEN station_inv.first_year AND station_inv.last_year
    GROUP BY 
      ob_date
  ) as sub_t 
  ON av_sites.ob_date = sub_t.ob_date 
  AND ST_Distance(av_sites.geog_point, station_inv.geog_point) = sub_t.min_distance
  ORDER BY av_sites.ob_date DESC;

