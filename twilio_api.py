import logging
from twilio.rest import Client
from twilio.base.exceptions import TwilioException


class TwilioSender:
    def __init__(self, filename):
        with open(filename) as f:
            account_sid, auth_token, from_phone_num, my_phone_num = f.readlines()
        account_sid = account_sid[:-1]
        auth_token = auth_token[:-1]
        self.from_phone_num = from_phone_num[:-1]
        self.my_phone_num = my_phone_num[:-1]
        self.client = Client(account_sid, auth_token)

    def send_message(self, msg):
        try:
            message = self.client.messages.create(body=msg, from_=self.from_phone_num, to=self.my_phone_num)
            logging.debug(message.sid)
        except TwilioException as ex:
            logging.warning(f"Twilio exception: {ex}")
