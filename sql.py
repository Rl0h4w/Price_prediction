import sqlalchemy

def main():
    engine = sqlalchemy.create_engine("mssql://user:password@85.114.8.250/qwerty")
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("""SELECT 
                YEAR(Date_time) as Год,
                MONTH(Date_time) as Месяц,
                DAY(Date_time) as День,
                Stations.Station_name,
                Stations.Gen_unit_name,
                AVG(Generation.Generation_hour) as Средняя_выработка_за_сутки,
                SUM(Generation.Generation_hour) as Суммарная_выработка_за_сутки
            FROM 
                Generation
            JOIN 
                Date ON Generation.Date_id = Date.Date_id
            JOIN 
                Stations ON Generation.Gen_unit_id = Stations.Gen_unit_id
            WHERE 
                YEAR(Date_time) = 2021
            GROUP BY 
                Date.Date_id, Stations.Station_name, Stations.Gen_unit_name"""))
        print(result.all())
if __name__ == "__main__":
    main()