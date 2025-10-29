docker compose -f docker-compose-base.yml --profile elasticsearch up es-kibana-setup
docker compose -f docker-compose-base.yml --profile elasticsearch up -d kibana
