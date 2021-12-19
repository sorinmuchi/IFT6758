# IFT6758 Milestone 3

In this milestone we're using Docker to deploy our Machine Learning models from previous Milestones.

## Run Entire Project
### Server Side and Jupyter Notebook
To run the project: `docker compose up`.

The first time this command is run it will also build the Docker images.

### Rebuild Container Images
If required to re-build the images: `docker compose build`.

## Convenience Scripts - Server-Only
To run the server-only:  `./build.sh && ./run.sh`.