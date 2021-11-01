--1-----------------------------------------------------------------------------
/*
Изучите таблицу airports и выведите список городов (city), в которых есть аэропорты.
Найдите уникальные значения.*/
SELECT 
    DISTINCT city
    
FROM airports;



--2----------------------------------------------------------------------------
/*
Изучите таблицу flights и подсчитайте количество вылетов (flight_id) из каждого аэропорта вылета (departure_airport). Назовите переменную cnt_flights и выведите её вместе со столбцом departure_airport — сначала departure_airport, потом cnt_flights. Результат отсортируйте в порядке убывания количества вылетов.
Примените команду GROUP BY*/
SELECT 
    departure_airport, -- ищем в аэропартах отправления
    COUNT(flight_id) AS cnt_flights -- считаем вылеты 
FROM flights -- из таблицы flight

GROUP BY departure_airport -- группируем по ааэропорту отправления
ORDER BY cnt_flights DESC; -- отсортируем в порядке убывания



--3-----------------------------------------------------------------------------
/*
Найдите количество рейсов на каждой модели самолёта с вылетом в сентябре 2018 года. Назовите получившийся столбец flights_amount и выведите его, также выведите столбец model.
Столбцы выводите в таком порядке:
model
flights_amount
Для решения задачи соедините таблицы aircrafts и flights. Используйте агрегирующие функции с группировкой и условие.*/
SELECT
    aircrafts.model AS model, -- выведим модели из таблицы aircrafts в столбец model
    COUNT(flights.flight_id) AS flights_amount -- посчитаем количество вылетов
FROM 
    flights
RIGHT JOIN aircrafts ON aircrafts.aircraft_code = flights.aircraft_code -- соединим  таблицы aircrafts и flights
WHERE EXTRACT(MONTH FROM departure_time) = '9' -- подзапрос
GROUP BY -- сгруппируем по model
    aircrafts.model;



--4-----------------------------------------------------------------------------
/*
Посчитайте количество рейсов по всем моделям самолётов Boeing, Airbus и другим ('other') в сентябре. Типы моделей поместите в столбец type_aircraft, а количество рейсов — во flights_amount. Выведите их на экран.
Соедините таблицы aircrafts и flights. Примените конструкцию CASE для группировки моделей.*/
SELECT
    CASE 
        WHEN aircrafts.model LIKE '%Boeing%' THEN 'Boeing'
        WHEN aircrafts.model LIKE '%Airbus%' THEN 'Airbus'
        ELSE 'other'
    END AS type_aircraft,

    COUNT(flights.flight_id) as flights_amount-- посчитаем количество вылетов    
        
FROM
    flights

    INNER JOIN aircrafts ON aircrafts.aircraft_code = flights.aircraft_code
--INNER JOIN flights ON aircrafts.aircraft_code = flights.aircraft_code

WHERE
    EXTRACT(MONTH FROM departure_time) = '9'
GROUP BY type_aircraft;



--5------------------------------------------------------------------------------
/*
Посчитайте среднее количество прибывающих рейсов в день для каждого города за август 2018 года. Назовите получившееся поле average_flights, вместе с ним выведите столбец city.
Выводите столбцы в таком порядке:
city,
average_flights.
Соедините таблицы flights и airports. Посчитайте количество рейсов для каждого города и найдите среднее функцией AVG().*/
SELECT
    fly_cnt.city AS city,
    AVG (flight_amount) AS average_flights
FROM(
    SELECT airports.city AS city,
    CAST(flights.arrival_time AS date) AS date_fly,
    COUNT (flights.arrival_time) AS flight_amount
    FROM 
        flights
    INNER JOIN airports ON airports.airport_code = flights.arrival_airport
    WHERE 
        EXTRACT(MONTH FROM flights.arrival_time) = '8'
    GROUP BY
        city,
        date_fly)
        AS fly_cnt
GROUP BY
    city;



--6--------------------------------------------------------------------------------
/*
Установите фестивали, которые проходили с 23 июля по 30 сентября 2018 года в Москве, и номер недели, в которую они проходили. Выведите название фестиваля festival_name и номер недели festival_week.
Для решения задачи используйте таблицу festivals.*/
SELECT
	festival_name,
    EXTRACT(week FROM festival_date) AS festival_week
FROM
    festivals
WHERE
    festival_city = 'Москва'
    AND 
    festival_date BETWEEN '2018-07-23' AND '2018-09-30';



--7--------------------------------------------------------------------------------
/*
Для каждой недели с 23 июля по 30 сентября 2018 года посчитайте количество билетов, купленных на рейсы в Москву (номер недели week_number и количество билетов ticket_amount). Получите таблицу, в которой будет номер недели; информация о количестве купленных за неделю билетов; номер недели ещё раз, если в эту неделю проходил фестиваль, и nan, если не проходил; а также название фестиваля festival_name.
Выводите столбцы в таком порядке:
week_number,
ticket_amount,
festival_week,
festival_name.
*/
SELECT
    week_number,
    ticket_amount,
    festival_week,
    festival_name

FROM
    (SELECT 
        EXTRACT(week FROM arrival_time) AS week_number,
        COUNT(ticket_no) AS ticket_amount
    FROM 
        flights
    LEFT JOIN airports ON airports.airport_code = flights.arrival_airport
    LEFT JOIN ticket_flights ON ticket_flights.flight_id  = flights.flight_id
    WHERE 
        CAST(arrival_time AS date) <= '2018-09-30' AND
        CAST(arrival_time AS date) >= '2018-07-23' AND
        city = 'Москва'
    GROUP BY
        week_number)
AS flights
LEFT JOIN 
    (SELECT
    festival_name,
    EXTRACT (WEEK FROM festival_date) AS festival_week
    FROM
        festivals
    WHERE
        CAST(festival_date AS date) <= '2018-09-30' AND
        CAST(festival_date AS date) >= '2018-07-23' AND
        festival_city = 'Москва')
AS FEST
ON FEST.festival_week = flights.week_number
    ORDER BY
        week_number;




