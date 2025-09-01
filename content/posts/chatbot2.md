+++
authors = ["Mike Harmon"]
title = "Building & Deploying A Serverless Multimodal ChatBot: Part 2"
date = "2025-01-09"
tags = [
    "LLMs",
    "Docker",
    "Google Cloud",
    "GitHub Actions",
    "CI/CD"
]
series = ["LLMs"]
aliases = ["migrate-from-jekyl"]
+++

### Contents
------------

__[1. Introduction](#first-bullet)__

__[2. Docker & Docker Hub](#second-bullet)__

__[3. GitHub Actions For CI/CD](#third-bullet)__

__[4. Deploying On Google Cloud Run](#fourth-bullet)__

__[5. Conclusions](#fifth-bullet)__


### Introduction <a class="anchor" id="first-bullet"></a>
----------------
In my [last post](https://mdh266.github.io/posts/chatbot1/) I went over how to create a create speech based chatbot app with a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) using [LangChain](https://www.langchain.com/), [Llama 3](https://ai.meta.com/blog/meta-llama-3/), [Google Cloud API]() and [Streamlit](https://streamlit.io/).

 In this post I'll cover how to deploy this app using [Docker](https://www.docker.com/) for containerization. Containerizing the app will allow us to run it both locally and on the cloud. Then I'll cover [GitHub Actions](https://github.com/features/actions) for automatically building the image and pushing it to [Docker Hub](https://hub.docker.com/) where it can be pulled and run on [Google Cloud Run](https://cloud.google.com/run) to create a serverless application.

### Docker & DockerHub <a class="anchor" id="second-bullet"></a>
------------------

[Docker](https://www.docker.com/) is the industry standard when it comes to containarizing applications. Containerization has made deploying applications and maintaining them across different environments much easier! Once a container is running on one computer it runs on all computers with Docker installed on it.

The three things you should know about Docker are: [images, containers](https://aws.amazon.com/compare/the-difference-between-docker-images-and-containers/) and [Dockerfiles](https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/).

A Docker image is a blue-print for a Docker container. A container is the instantiation of that image. This is similar to the way an object is an instantiation of a class in object oriented programing. The definition of an image is given by the Dockerfile. 

The Dockerfile for this project is pretty simple:

    FROM python:3.11-slim

    RUN mkdir /app
    RUN mkdir /app/src
    WORKDIR /app

    COPY src /app/src
    COPY pyproject.toml /app
    COPY entrypoint.sh /app
    RUN chmod +x /app/entrypoint.sh
    RUN pip install . --no-cache 

    ENTRYPOINT ["/app/entrypoint.sh"]

The two things that I will point out that are a little different is the use of `--no-cache` for pip installing our Python dependencies. By default pip stores the download packages in a cached directory so that subsequent installations of the same package will be faster.  However, since we dont need re-install anything in the container and the packages can take up signficant space (causing the images to bloat) I avoid caching them. Doing so made my image to be 618MB while with caching packages the image was 734MB. 100MB may not seem like much, but it's almost 20% larger with caching and in prior projects the images I made used tons of packages with caching and ended up using GBs.

The second point I'll call out is the use of the [entrypoint.sh](https://github.com/mdh266/thirdapp/blob/main/entrypoint.sh) script. For some reason it was not possible to run Streamlit on Google Cloud Run using the `streamlit run ...` command directly in the `ENTRYPOINT`, but invoking a Bash script  with that command in it did work and that's the reason for it!

The Docker image can be built from the Dockerfile using the command,

    docker image build -t <image_name> .

The container can be run using the command,

    docker run -ip 8080:8080 -e GROQ_API_KEY=<your-groq-api> -e GOOGLE_API_KEY=<your-google-api>

Notice I had to use `-ip 8080:8080` to perform [port-forwarding](https://en.wikipedia.org/wiki/Port_forwarding) from the container to my machine. I used port 8080 instead of Streamlit's default port of 8051 since Google Cloud Run uses port 8080 and its easy enough to switch ports in Docker. I also pass the API keys in as environment variables to the container using the `-e` syntax. 

**NEVER load your `.env` file in your image or set your API keys in the image. If you do then anyone can get them when they get access to the image!**

It is a little bit annoying to have to pass this environment variables all the time, especially as you use more and more API keys, so for local development I used [Docker Compose](https://docs.docker.com/compose/) and the following [docker-compose.yml](https://github.com/mdh266/thirdapp/blob/main/docker-compose.yml) file,

    services:
        app:
            build: .
            env_file: ".env"
            ports:
            - "8080:8080"

Notice the `app` specifies building the image and the `env_file` variable specifies my ".env" file with my API keys. This is **okay** since Docker Compose will inject the environment variables into the container and not into the image! 

You can start up the container with the command,

    docker compose up

And then the site should be running on https://localhost:8080.

The last part to this section is a discussion of [Docker Hub](https://hub.docker.com/). Docker Hub is a repository used to host Docker images that can used to pull images from and run them on different platforms. I used Docker Hub to host the image so that I could pull it and run it from Google Cloud Run. The command to push the image to Docker Hub is,

    docker push mdh266/thirdapp:cloudrun

where `mdh266` is my Docker Hub account name, `thirdapp` is the name of the image and `cloudrun` is the tag for the version. One problem that I had was I used a M1 based Apple computer to build the image and had trouble running it on a Linux machine on Google. This is a [known problem](https://pythonspeed.com/articles/docker-build-problems-mac/) and I used this as an opportunity to build the image on a Linux machine using [GitHub Actions](https://github.com/features/actions).

### GitHub Actions For CI/CD <a class="anchor" id="third-bullet"></a>
------------------------------

[GitHub Actions](https://github.com/features/actions) is an easy way to integrate [CI/CD](https://en.wikipedia.org/wiki/CI/CD) natively with GitHub. I'll create a GitHub Action to build and push the Docker image for this project to DockerHub I'll create a [YAML](https://en.wikipedia.org/wiki/YAML) file called [docker-build.yaml](https://github.com/mdh266/speech-chatbot/blob/main/.github/workflows/docker-build.yaml) that has to be under the following (hidden) folder structure:

    speech-chatapp/
        .github/workflows/docker-build.yaml


This action will run on the any push to the main branch. On action kicks off a job that is defined as,

    jobs:
        build-docker:
            runs-on: ubuntu-latest
            steps:
            - name: checkout
                uses: actions/checkout@v3
            
            - name: Login to Docker Hub
                uses: docker/login-action@v3
                with:
                username: ${{ secrets.DOCKERHUB_USERNAME }}
                password: ${{ secrets.DOCKERHUB_TOKEN }}

            - name: Build & Push Docker
                run: | 
                docker build -t mdh266/thirdapp:cloudrun .
                docker push mdh266/thirdapp:cloudrun 

 The `docker` job runs on an ubuntu based machine and run steps that checkout the code, signs into to Docker Hub, builds and pushes the image to Docker Hub. 
 
 Notice that I have to use variables `${{ secrets.DOCKERHUB_USERNAME }}` and `${{ secrets.DOCKERHUB_TOKEN }}` that hold my Docker Hub user name and Docker Hub API key. In order to set these variables I have to add their secrets to the GitHub repo. I click on "Settings" for the repo then on the bottom left panel I click on "Secrets and Variables" drop down. Then I click on "Actions" as shown below,


<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/3_github_secrets.png?raw=1">
</figure>

 I can click on the green "New repository secret" and then add the `DOCKERHUB_USERNAME` with the value shown below,

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/2_addghsecret.png?raw=1">
</figure>


Now do the same for the `DOCKERHUB_TOKERN` and then I am all set up so that any time I push to the main branch it will rebuild the Docker image and push it to Docker Hub. The last step is to deploy the container on Google Cloud Run.


### Deploying On Google Cloud Run <a class="anchor" id="fourth-bullet"></a>
----------------------------------

Deploying on app on [Google Cloud Run](https://cloud.google.com/run?hl=en) is relatively straight forward. You can create an app by clicking "DEPLOY CONTAINER" as shown below,

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/0_deploy.jpg?raw=1">
</figure>

Then select "Service" and fill out the form as shown below,

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/5_CreateApp.png?raw=1">
</figure>

Notice that "Container image URL" matches the image name that I pushed to on Docker Hub above. Scrolling down we can see the settings I made for the number of instances to scale down to zero (to reduce cost). To allow all anyone to access the app I set "Authentication" to "Allow unathenticated invocations" and "Ingress Control" to "All" , 

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/6_AllTraffic.png?raw=1">
</figure>

The last step is to click on the "Container(s), Volume, Networks, Security" drop down and then set the API keys as Environment Variables as shown below,

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/8_API_KEYS.png?raw=1">
</figure>

We can finally hit "Create" at the very bottom and we are done!

One extra is that if I wanted to create an actual website url for this app I can use [Domain NameSpace (DNS) Mapping](https://cloud.google.com/appengine/docs/legacy/standard/python/mapping-custom-domains). In order to do so I followed the steps laid out in [this youtube video](https://www.youtube.com/watch?v=lDtvpUYAFzA).


Lastly I can fully automate the building and deploying of this app by adding Google Cloud Run deployment to GitHub Actions. To learn how to do this I used this [YouTube video](https://www.youtube.com/watch?v=kZYsoav104w) and its associated [GitHub](https://github.com/thaddavis/how-to-deploy-a-dockerized-fastapi-to-google-cloud-run/blob/main/.github/workflows/cicd.yaml). The first step was to create a Service Account as shown below,


<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/create_sa1.png?raw=1">
</figure>

Then give the service a name,

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/create_sa2.png?raw=1" width="500"/>
</figure>

Then add these roles (I was stuck at this part until I found this [stackoverflow post](https://stackoverflow.com/questions/55788714/deploying-to-cloud-run-with-a-custom-service-account-failed-with-iam-serviceacco
) ),

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/roles.png?raw=1" width="500"/>
</figure>

Now you can click Continue. After you create the account you create a new key associated with it and download the json credentials,

<figure>
<img src="https://github.com/mdh266/speech-chatbot/blob/main/notebooks/images/keys.png?raw=1" width="500"/>
</figure>


I added this json file to GitHub sercrets as `GOOGLE_CREDENTIALS`. I also added `GOOGLE_PROJECT_ID`, my `GROQ_API_KEY` and my `GOOGLE_API_KEY` to GitHub secrets as well. I then added the following to steps to my GitHub Actions YAML:

      - name: Google Cloud Auth
        uses: 'google-github-actions/auth@v2'
        with:
          credentials_json: '${{ secrets.GOOGLE_CREDENTIALS }}'
  
      - name: Set up Cloud SDK
        uses: 'google-github-actions/setup-gcloud@v2'
  
      - name: Deploy to Cloud Run
        run: |
            gcloud run deploy $SERVICE_NAME \
            --image=${{ env.DOCKER_IMAGE }}:cloudrun \
            --set-env-vars=GROQ_API_KEY=${{ secrets.GROQ_API_KEY }},GOOGLE_API_KEY=${{ secrets.GOOGLE_API_KEY }} \
            --region=us-central1 \
            --project=${{ secrets.GOOGLE_PROJECT_ID }} 


The environment variables `$SERVICE_NAME` and `DOCKER_IMAGE` were set to `mikegpt` and `mdh266/thirdapp` respectively earlier in my [YAML file](https://github.com/mdh266/speech-chatbot/blob/main/.github/workflows/docker-build.yaml#L4).

Now I am done! Every time I push to the main branch on GitHub the Docker image will be built, pushed to Docker Hub then pulled from Docker Hub and re-deployed on Google Cloud Run!

### Conclusions <a class="anchor" id="fifth-bullet"></a>
---------------------------

In this post I went over a lot, but the steps are pretty straight forward. I covered how to build Docker images and containers. Next I covered how to set up GitHub actions to automatically build and push the Docker image to Docker Hub. Lastly, I showed how to pull that image and run it as a container on Google Cloud Run and automate the whole process as a GitHub Action.

I hope you enjoyed this!
