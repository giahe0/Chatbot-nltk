from flask_wtf import FlaskForm
from wtforms import StringField,TextAreaField,SubmitField
from wtforms.validators import DataRequired
import googlesearch
class chatbotform(FlaskForm):
    user_input=StringField(validators=[DataRequired()])
    send=SubmitField('Send')
