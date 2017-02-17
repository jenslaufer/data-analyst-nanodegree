
# coding=utf-8
import xml.etree.cElementTree as ET
import re
import codecs
import json
import phonenumbers
from collections import defaultdict


STREET_TYPE_RE = re.compile(r'\b\S+\.?$', re.IGNORECASE)
LOWER = re.compile(r'^([a-z]|_)*$')
LOWER_COLON = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
URL_RE = re.compile(r'^(https?:\/\/)([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$')
EMAIL_RE = re.compile(r'^([\u00C0-\u017a-zA-Z\d_\.-]+)@([\u00C0-\u017\da-zA-Z\.-]+)\.([a-z\.]{2,6})$')

def count_tags(filename):
    tags = {}
    for event, elem in ET.iterparse(filename, events=("start",)):
        num = 1
        try:
            num = tags[elem.tag]
            num += 1
        except:
            pass
        tags[elem.tag] = num 
    return tags



def key_type(element, keys, problematic, other):
    if element.tag == "tag":
        key = element.attrib['k']
        if LOWER.match(key):
            keys["LOWER"] = keys["LOWER"] + 1
        elif LOWER_COLON.match(key):
            keys["LOWER_COLON"] = keys["LOWER_COLON"] + 1
        elif PROBLEMCHARS.match(key):
            keys["PROBLEMCHARS"] = keys["PROBLEMCHARS"] + 1
            problematic.append(key)
        else:
            keys["other"] = keys["other"] + 1
            other.append(key)

    return keys,problematic,other


def audit_k_value(filename):
    keys = {"LOWER": 0, "LOWER_COLON": 0, "PROBLEMCHARS": 0, "other": 0}
    problematic = []
    other = []
    for _, element in ET.iterparse(filename):
        keys,problematic,other = key_type(element, keys, problematic, other)

    return keys,problematic,other 




def contributing_users(filename):
    users = {}
    for _, element in ET.iterparse(filename):
        if element.tag == 'node':
            uid = element.attrib['uid']
            name = element.attrib['user']
            users[uid] = name

    return users


def is_valid_email(email):
    return EMAIL_RE.match(email)

def is_valid_url(url):
    return URL_RE.match(url)


def is_valid_phone(phone):
    try:
        return phonenumbers.is_valid_number(phonenumbers.parse(phone))
    except:
        return False


def clean_contact(contact):
    return contact


def clean_address(address):
    return address


def audit_contact_data(filename):
    invalid_phone = []
    invalid_email = []
    invalid_url = []
    with open(filename, 'rb') as f:
        for event, elem in ET.iterparse(f, events=("start",)):

            if elem.tag == "node" or elem.tag == "way":
                for tag in elem.iter("tag"):
                    if tag.get('k') == "contact:website":
                        if not is_valid_url(tag.get('v')):
                            invalid_url.append(tag.get('v'))
                    elif tag.get('k') == "contact:phone":
                        if not is_valid_phone(tag.get('v')):
                            invalid_phone.append(tag.get('v'))
                    elif tag.get('k') == "contact:fax":
                        if not is_valid_phone(tag.get('v')):  
                            invalid_phone.append(tag.get('v'))
                    elif tag.get('k') == "contact:email":
                        if not is_valid_email(tag.get('v')):
                            invalid_email.append(tag.get('v'))

    return (invalid_phone, invalid_email, invalid_url)

def clean_node(node):
    try:
        node['address'] = clean_address(node['address'])
    except:
        pass

    try:
        node['contact'] = clean_contact(node['contact'])
    except:
        pass

    return node



def shape_element(element):


    CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

    node = {}
    if element.tag == "node" or element.tag == "way" :
        node['pos'] = [0,0]
        node['created'] = {}
        node['type'] = element.tag
        if element.get('id') != None:
            node['id'] = element.get('id')
        if element.get('visible') != None:
            node['visible'] = element.get('visible')
        if element.get('version') != None:
            node['created']['version'] = element.get('version')
        if element.get('changeset') != None:
            node['created']['changeset'] = element.get('changeset')
        if element.get('timestamp') != None:
            node['created']['timestamp'] = element.get('timestamp')
        if element.get('user') != None:
            node['created']['user'] = element.get('user')
        if element.get('uid') != None:
            node['created']['uid'] = element.get('uid')
        if element.get('lat') != None:
            node['pos'][0] = float(element.get('lat'))
        if element.get('lon') != None:
            node['pos'][1] = float(element.get('lon'))
        
        address = {}
        contact = {}
        for child in element:
            if child.tag == 'tag':
                if child.attrib['k'] == 'addr:housenumber':
                    address['housenumber'] = child.get('v')
                elif child.attrib['k'] == 'addr:country':
                    address['country'] = child.get('v')
                elif child.attrib['k'] == 'addr:postcode':
                    address['postcode'] = child.get('v')
                elif child.attrib['k'] == 'addr:street':
                    address['street'] = child.get('v')
                elif child.attrib['k'] == 'addr:city':
                    address['city'] = child.get('v')
                elif child.attrib['k'] == 'contact:email':
                    contact['email'] = child.get('v')
                elif child.attrib['k'] == 'contact:fax':
                    contact['fax'] = child.get('v')
                elif child.attrib['k'] == 'contact:phone':
                    contact['phone'] = child.get('v')
                elif child.attrib['k'] == 'contact:website':
                    contact['website'] = child.get('v')
                elif child.attrib['k'] == 'amenity':
                    node['amenity'] = child.get('v')
                elif child.attrib['k'] == 'name':
                    node['name'] = child.get('v') 
                elif child.attrib['k'] == 'tourism':
                    node['tourism'] = child.get('v') 

        if len(address) > 0:
            node['address'] = address
        if len(contact) > 0:
            node['contact'] = contact
        return clean_node(node)
    else:
        return None


def process(file_in, file_out, pretty = False):
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

