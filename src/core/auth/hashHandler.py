from bcrypt import hashpw,checkpw,gensalt

class HashHelper(object):
    @staticmethod
    def verify_password(password:str,hashedPassword:str):
        if checkpw(password.encode("utf-8"),hashedPassword.encode("utf-8")):
            return True
        else:
            return False
        
    @staticmethod
    def get_password_hash(password:str):
        return hashpw(
            password.encode('utf-8'),
            gensalt()
        ).decode('utf-8')