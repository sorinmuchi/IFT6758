# IFT6758 Milestone 3

In this milestone we're using Docker to deploy our Machine Learning models from previous Milestones.

## Run Entire Project
### Setup
1. Install Docker
2. Create a `.env` file using the template shown in the `.env.example` file.
3. In `.env` add a valid API key for the `COMET_API_KEY` environment variable, i.e. replace the `____REPLACE_ME____` value with an API key value obtained from the [comet.ml model repository](https://www.comet.ml/meriembchaaben/ift6758).

### Start Server Side and Jupyter Notebook
To run the project: `docker compose up`.

The first time this command is run it will also build the Docker images.

### Rebuild Container Images
If required to re-build the images: `docker compose build`.

## Convenience Testing Scripts - Server-Only
To run the server-only:  `./build.sh && ./run.sh`.
