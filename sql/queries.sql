-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS qualifying_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: products above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers with no completed order
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: customers above average completed spend
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed'
    GROUP BY customer_id
), average_spend AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend
FROM customer_spend cs CROSS JOIN average_spend a
WHERE cs.total_spend > a.avg_spend;

-- Q6: within-city ranking of completed spend
WITH customer_spend AS (
    SELECT c.city, c.customer_id, SUM(o.order_amount) AS total_spend
    FROM customers c JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;

-- Q7: daily revenue and cumulative revenue
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue_inr,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue_inr
FROM daily ORDER BY day;
