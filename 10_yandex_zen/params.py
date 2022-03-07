#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import getopt


if __name__ == '__main__':

    # Задаём формат входных параметров
    unixOptions = 's:e'  
    gnuOptions = ['start_dt=', 'end_dt=']

    # Получаем строку входных параметров
    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]

    # Проверяем входные параметры на соответствие формату,
    # заданному в unixOptions и gnuOptions
    try:  
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:  
        print (str(err))
        sys.exit(2)      # Прерываем выполнение, если входные параметры некорректны

    # Считываем значения из строки входных параметров
    start_dt = ''
    end_dt = ''   
    for currentArgument, currentValue in arguments:  
        if currentArgument in ('-s', '--start_dt'):
            start_dt = currentValue                                   
        elif currentArgument in ('-e', '--end_dt'):
            end_dt = currentValue        

    # Выводим результат
    print(start_dt, end_dt)