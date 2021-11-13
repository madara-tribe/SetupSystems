local2ec2:  # folder
        scp -r -i ~/.ssh/hagis.pem BERT ubuntu@(AD):/home/ubuntu/BERT

ec2local: # folder
        scp -r -i ~/.ssh/hagis.pem ubuntu@{AD}:/home/ubuntu/BERT /Users/hagiharatatsuya/downloads/

whenfile:
        scp -i ~/.ssh/hagis.pem
