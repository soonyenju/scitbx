#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Songyan Zhu
# Mail: zhusy93@gmail.com
# Created Time:  2018-10-23 13:28:34
#############################################


from setuptools import setup, find_packages

setup(
	name = "scitbx",
	version = "0.0.92",
	keywords = ("Geosciences, geospatial, data science"),
	description = "For academic data processing and plotting etc.",
	long_description = "Processing academic data and drawing figures for geosciences researchers.",
	license = "MIT Licence",

	url="https://github.com/soonyenju/scitbx",
	author = "Songyan Zhu",
	author_email = "zhusy93@gmail.com",

	packages = find_packages(),
	include_package_data = True,
	platforms = "any",
	install_requires=[

	]
)