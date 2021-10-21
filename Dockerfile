FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=America/Chicago

# Add zsh to container
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting

#RUN apt-get update -qq
#RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends libgl1-mesa-glx ffmpeg \
#    libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -qq

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt

#WORKDIR /workspace

#RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
#ENTRYPOINT ["/entry.sh"]
