#!/bin/sh
docker run -it -p 6758:6758 --env-file .env ift6758/serving:1.0.0
