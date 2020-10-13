# Bike Sharing Project

In this project, I developed a bike share prediction, which is a submission project of Udacity Deep Learning Nanodegree Program, with data and part of the code provided in advance by Udacity.

<b>Mission:</b> Predicting from historical data, how many bike it'll be needed in the future.

<b>Features:</b> 

<ul>
<li><b>instant</b>     : ID</li>
<li><b>dteday</b>     : Date in YYYY-MM-DD format</li>
<li><b>season</b>     : Season (1:Springer, 2:Summer, 3:Fall, 4:Winter)</li>
<li><b>yr</b>         : Year (0: 2011, 1:2012)</li>
<li><b>mnth</b>       : Month (1 to 12)</li>
<li><b>hr</b>         : Hour (0 to 24)</li>
<li><b>holiday</b>    : Weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)</li>
<li><b>weekday</b>    : Day of the week</li>
<li><b>workingday</b> : If day is neither weekend nor holiday is 1, otherwise is 0.</li>
<li><b>weathersit</b> :</li>
        <ul>
            <li><b>1</b>: Clear, Few clouds, Partly cloudy, Partly cloudy</li>
            <li><b>2</b>: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist</li>
            <li><b>3</b>: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds</li>
            <li><b>4</b>: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog</li>
        </ul>
<li><b>temp</b>       : Normalized temperature in Celsius. The values are divided to 41 (max)</li>
<li><b>atemp</b>      : Normalized feeling temperature in Celsius. The values are divided to 50 (max)</li>
<li><b>hum</b>        : Normalized humidity. The values are divided to 100 (max)</li>
<li><b>windspeed</b>  : Normalized wind speed. The values are divided to 67 (max)</li>
<li><b>casual</b>     : Count of casual users</li>
<li><b>registered</b> : Count of registered users</li>
<li><b>cnt</b>        : Count of total rental bikes including both casual and registered</li>
</ul>
