FROM public.ecr.aws/lambda/python:3.10



RUN yum install -y \
    gcc \
    python3-devel \
    && rm -rf /var/cache/yum/*

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install netifaces --no-binary :all:
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" -U --no-cache-dir

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]

