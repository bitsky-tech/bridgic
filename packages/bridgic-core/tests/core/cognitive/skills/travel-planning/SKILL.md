---
name: travel-planning
description: "MUST USE when planning trips, booking flights, or reserving hotels. Request skill details FIRST before executing travel-related tasks to get the standard workflow."
---

# Travel Planning Skill

This skill provides structured guidance for completing travel planning tasks.

## Standard Travel Planning Process

When helping users plan travel, follow these typical steps in sequence:

### Step 1: Gather Flight Information
- **What to do**: Find available flight options between the origin and destination cities
- **What you need**: Departure city, destination city, and travel date
- **What to expect**: A list of flight options with details like flight numbers, departure times, and prices
- **How to proceed**: If the user hasn't specified a date, use a reasonable default (e.g., tomorrow or the next available date) and proceed with gathering flight information

### Step 2: Complete Flight Booking
- **What to do**: Reserve a flight from the available options
- **What you need**: A specific flight selection (prefer the most economical option if multiple choices exist)
- **What to expect**: Confirmation of the flight booking with booking details
- **How to proceed**: After reviewing flight options, select the most suitable one (typically the cheapest) and proceed with booking

### Step 3: Gather Accommodation Information
- **What to do**: Find available hotels or accommodations at the destination
- **What you need**: Destination city, check-in date, and check-out date
- **What to expect**: A list of accommodation options with details like hotel names, prices, and amenities
- **How to proceed**: Use the flight arrival date as check-in date, and estimate a reasonable stay duration (e.g., 2-3 days) if not specified

### Step 4: Complete Accommodation Booking
- **What to do**: Reserve a hotel or room from the available options
- **What you need**: A specific accommodation selection (prefer the most economical option if multiple choices exist)
- **What to expect**: Confirmation of the accommodation booking with reservation details
- **How to proceed**: After reviewing accommodation options, select the most suitable one (typically the cheapest) and proceed with booking

## Key Principles

1. **Proactive execution**: Take initiative to gather information and make reasonable assumptions when details are missing (e.g., use tomorrow's date if not specified, choose economical options)
2. **Sequential flow**: Each step naturally follows from the previous one - flight information leads to booking, which informs accommodation dates
3. **Information extraction**: Extract key details (flight numbers, hotel names, dates) from previous step results to use in subsequent steps
4. **Complete the task**: Don't stop at gathering information - proceed to complete bookings to fully satisfy the user's travel planning needs
