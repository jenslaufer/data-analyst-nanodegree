
# coding=utf-8
import xml.etree.cElementTree as ET
import re
import codecs
import json
from collections import defaultdict


STREET_TYPE_RE = re.compile(r'\b\S+\.?$', re.IGNORECASE)
LOWER = re.compile(r'^([a-z]|_)*$')
LOWER_COLON = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

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



def key_type(element, keys):
    if element.tag == "tag":
        key = element.attrib['k']
        if LOWER.match(key):
            keys["LOWER"] = keys["LOWER"] + 1
        elif LOWER_COLON.match(key):
            keys["LOWER_COLON"] = keys["LOWER_COLON"] + 1
        elif PROBLEMCHARS.match(key):
            keys["PROBLEMCHARS"] = keys["PROBLEMCHARS"] + 1
        else:
            keys["other"] = keys["other"] + 1

    return keys


def audit_k_value(filename):
    keys = {"LOWER": 0, "LOWER_COLON": 0, "PROBLEMCHARS": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys


def contributing_users(filename):
    users = {}
    for _, element in ET.iterparse(filename):
        if element.tag == 'node':
            uid = element.attrib['uid']
            name = element.attrib['user']
            users[uid] = name

    return users


def audit_street_type(street_types, street_name):
    expected = ["Strasse", "Allee", "Straße", 'Weg', 'Ring']


    m = STREET_TYPE_RE.search(street_name)
    if m:
        street_type = m.group()
        if street_type.encode('utf-8') not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit_streets(osmfile):
    osm_file = open(osmfile, "rb")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.get('v'))
    osm_file.close()
    return street_types


def update_name(name, mapping):

    mapping = { "Str": "Straße",
            "St.": "Straße"
          }
    m = STREET_TYPE_RE.search(name)
    if m:
        street_type = m.group()
        return name.replace(street_type,mapping[street_type])
    else:
        return name





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
            node['pos'][0] = element.get('lat')
        if element.get('lon') != None:
            node['pos'][1] = element.get('lon')
        
        
        node['address'] = {}
        for child in element:
            if child.tag == 'tag':
                if child.attrib['k'] == 'addr:housenumber':
                    node['address']['housenumber'] = child.get('v')
                elif child.attrib['k'] == 'addr:postcode':
                    node['address']['postcode'] = child.get('v')
                elif child.attrib['k'] == 'addr:street':
                    node['address']['street'] = child.get('v')
                elif child.attrib['k'] == 'amenity':
                    node['amenity'] = child.get('v')
                elif child.attrib['k'] == 'name':
                    node['name'] = child.get('v')
                elif child.attrib['k'] == 'phone':       
                    node['phone'] = child.get('v')   
        return node
    else:
        return None


def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
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

