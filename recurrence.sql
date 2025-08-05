WITH ClientCombinations AS (
    -- Étape 1 : Pour chaque client, créer sa combinaison de règles et compter ses alertes
    SELECT
        id_client,
        -- Crée une chaîne triée des règles uniques pour garantir que 'A/B' et 'B/A' soient identiques
        STRING_AGG(DISTINCT règle, '/' ORDER BY règle) AS combinaison_regles,
        COUNT(id_alerte) AS total_alertes_client
    FROM
        alertes
    GROUP BY
        id_client
)
-- Étape 2 : Compter les clients et les alertes pour chaque combinaison
SELECT
    combinaison_regles,
    COUNT(id_client) AS nombre_clients_uniques,
    SUM(total_alertes_client) AS nombre_total_alertes
FROM
    ClientCombinations
GROUP BY
    combinaison_regles
ORDER BY
    nombre_clients_uniques DESC;
